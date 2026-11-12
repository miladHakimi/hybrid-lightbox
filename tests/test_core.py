import numpy as np
from PIL import Image

import lightbox.cli as cli
from lightbox.model import Project, Layer, LayerMode, Material, SourceImage, luminance, magic_wand_region
from lightbox.optics import (OPTIMIZER_NOZZLE_DIAMETER_MM, thickness_map,
                             combined_transmission, optimize_rear_openings,
                             layer_thickness, nozzle_resolved_values, preview_u8)
from lightbox.geometry import voxel_surface, is_watertight, connected_components, write_ascii_stl, write_binary_stl, add_frame_and_bridges, add_hidden_bridges, remove_small_features, smooth_print_data
from lightbox.cli import DEFAULT_CONNECTOR_DEPTH_MM, DEFAULT_STL_DIMENSIONS_MM, discover_images, export, generate_one, mesh_inputs, parse_points, reinforce_connectors, reinforce_frame, scale_mesh_to_dimensions
from lightbox.ai import SAM_MODELS, LocalAI, assign_depth_layers, assign_prompted_layers, apply_ai_decomposition, filter_segments, normalize_depth, select_ai_device
from lightbox.view import ConnectorCache, automatic_view, brush_cursor_diameter, connector_masks, connector_view_preview, image_view_preview, source_image_preview, supports_drag_paint, valid_segment_indices, view_for_tool
from lightbox.keybindings import COMMANDS, DEFAULT_BINDINGS, KeybindingStore, dispatch_keybinding


def project(mask=None):
    image=np.full((3,4,3),128,np.uint8); mask=np.ones((3,4),bool) if mask is None else mask
    return Project(image,[Layer("front",mask.copy(),LayerMode.LITHOPHANE,Material(.8,3,.7)),
                          Layer("middle",mask.copy(),LayerMode.HYBRID,Material(.8,3,.7)),
                          Layer("rear",mask.copy(),LayerMode.SILHOUETTE,Material(.8,3,.7))],120)


def test_layer_settings_update_selected_layer_or_all_layers_independently():
    p=project();settings=Material(.5,2.5,1.1,.9,True)
    changed=p.update_layer_settings(1,LayerMode.LITHOPHANE,settings)
    assert changed==[1]
    assert p.layers[1].mode==LayerMode.LITHOPHANE and p.layers[1].material==settings
    assert p.layers[0].mode==LayerMode.LITHOPHANE and p.layers[0].material.minimum_mm==.8

    changed=p.update_layer_settings(1,LayerMode.HYBRID,settings,apply_to_all=True)
    assert changed==[0,1,2]
    assert all(layer.mode==LayerMode.HYBRID and layer.material==settings for layer in p.layers)
    assert all(layer.material is not settings for layer in p.layers)
    assert len({id(layer.material) for layer in p.layers})==3


def test_ai_uses_cpu_by_default_to_avoid_native_metal_command_buffer_crashes():
    class MPS:
        @staticmethod
        def is_available():return True
    class Backends:mps=MPS()
    class Torch:backends=Backends()
    assert select_ai_device(Torch(),platform_name="darwin")==-1
    assert select_ai_device(Torch(),requested="mps",platform_name="darwin")=="mps"


def test_keybinding_store_creates_an_empty_default_file(tmp_path):
    path=tmp_path/"keybindings.json"
    assert KeybindingStore(path).load()=={}
    assert path.exists()
    assert '"bindings": {}' in path.read_text()


def test_keybindings_reload_and_ignore_unknown_commands(tmp_path):
    path=tmp_path/"keybindings.json";store=KeybindingStore(path)
    store.save({"history.undo":"Alt+U","history.redo":"Alt+R","tool.brush":"B","view.image":"Ctrl+1","unknown.command":"X"})
    assert KeybindingStore(path).load()=={"history.undo":"Alt+U","history.redo":"Alt+R","tool.brush":"B","view.image":"Ctrl+1"}


def test_keybinding_commands_cover_tools_views_and_segment_edits():
    command_ids={command_id for command_id,_label in COMMANDS}
    assert {"tool.brush","tool.wand","tool.transform","tool.pen","segment.add","segment.erase"}<=command_ids
    assert {command_id for command_id in command_ids if command_id.startswith("view.")}


def test_keybinding_commands_include_undo_and_redo_with_standard_defaults():
    command_ids={command_id for command_id,_label in COMMANDS}
    assert {"history.undo","history.redo"}<=command_ids
    assert DEFAULT_BINDINGS=={"history.undo":"Ctrl+Z","history.redo":"Ctrl+Shift+Z"}


def test_keybinding_commands_dispatch_undo_and_redo():
    history=[]
    dispatch_keybinding("history.undo",lambda _value:None,lambda _value:None,lambda _value:None,lambda:history.append("undo"),lambda:history.append("redo"))
    dispatch_keybinding("history.redo",lambda _value:None,lambda _value:None,lambda _value:None,lambda:history.append("undo"),lambda:history.append("redo"))
    assert history==["undo","redo"]


def test_keybinding_commands_dispatch_to_editor_controls():
    tools=[];views=[];segment_operations=[]
    dispatch_keybinding("tool.wand",tools.append,views.append,segment_operations.append)
    dispatch_keybinding("tool.transform",tools.append,views.append,segment_operations.append)
    dispatch_keybinding("tool.pen",tools.append,views.append,segment_operations.append)
    dispatch_keybinding("view.backlit",tools.append,views.append,segment_operations.append)
    dispatch_keybinding("segment.add",tools.append,views.append,segment_operations.append)
    dispatch_keybinding("segment.erase",tools.append,views.append,segment_operations.append)
    assert tools==["Magic wand","Transform","Pen"]
    assert views==["Backlit preview"]
    assert segment_operations==["add","erase"]


def test_image_view_preview_shows_brush_editing_mask_over_source_image():
    image=np.full((3,4,3),100,np.uint8)
    mask=np.zeros((3,4),bool);mask[1,2]=True
    preview=image_view_preview(image,mask)
    assert np.array_equal(preview[0,0],image[0,0])
    assert not np.array_equal(preview[1,2],image[1,2])
    assert np.array_equal(image,np.full((3,4,3),100,np.uint8))


def test_source_image_preview_never_overlays_a_mask_from_another_image():
    image=np.full((3,4,3),100,np.uint8);mask=np.ones((3,4),bool)
    assert np.array_equal(source_image_preview(image,mask,1,0),image)
    assert not np.array_equal(source_image_preview(image,mask,1,1),image)


def test_connector_preview_generates_structure_without_changing_layer_masks():
    image=np.full((12,14,3),100,np.uint8)
    floating=np.zeros((12,14),bool);floating[5:8,6:9]=True
    p=Project(image,[Layer("floating",floating.copy())],width_mm=5.6)
    before=p.layers[0].mask.copy()
    additions=connector_masks(p)
    assert len(additions)==1 and additions[0].any()
    assert np.array_equal(p.layers[0].mask,before)
    assert len(connected_components(before|additions[0]))==1
    preview=connector_view_preview(image,before,additions[0])
    assert preview.dtype==np.uint8 and preview.shape==image.shape
    assert not np.array_equal(preview[additions[0]][0],preview[before][0])


def test_connector_cache_reuses_results_until_invalidated():
    calls=[]
    cache=ConnectorCache(lambda value: calls.append(value) or [value])
    first=cache.get("project")
    assert cache.get("project") is first
    assert calls==["project"]
    cache.invalidate()
    second=cache.get("changed project")
    assert second is not first
    assert calls==["project","changed project"]


def test_automatic_view_changes_preserve_connector_view():
    assert automatic_view("Connector view","Layer segments")=="Connector view"
    assert automatic_view("Connector view","Selected segment")=="Connector view"
    assert automatic_view("Image","Layer segments")=="Layer segments"


def test_leaving_transform_restores_source_editing_view_for_brush():
    assert view_for_tool("Image","Image","Transform")=="Selected mask"
    assert view_for_tool("Selected mask","Image","Brush")=="Image"
    assert view_for_tool("Selected mask","Selected segment","Magic wand")=="Selected segment"
    assert view_for_tool("Selected mask","Layer segments","Brush")=="Image"
    assert view_for_tool("Selected mask","Backlit preview","Brush")=="Image"


def test_brush_and_pen_support_continuous_mouse_drag_painting():
    assert supports_drag_paint("brush")
    assert supports_drag_paint("pen")
    assert not supports_drag_paint("wand")
    assert not supports_drag_paint("transform")


def test_brush_cursor_matches_radius_and_canvas_zoom():
    assert brush_cursor_diameter(5,500,1000)==5
    assert brush_cursor_diameter(10,500,1000)==10
    assert brush_cursor_diameter(10,1000,1000)==20
    assert brush_cursor_diameter(1,0,0)==2


def test_stale_segment_indices_are_removed_when_switching_images():
    assert valid_segment_indices([3,0,1,3,-1],2)==[0,1]
    assert valid_segment_indices([0],0)==[]


def test_subtract_layer_removes_overlap_after_both_output_transforms():
    image=np.zeros((7,7,3),np.uint8)
    target=np.zeros((7,7),bool);target[3,1:4]=True
    source=np.zeros((7,7),bool);source[3,1]=True
    p=Project(image,[Layer("target",target,offset_x=1),Layer("source",source,offset_x=3)])
    target_before=p.layers[0].mask.copy();source_before=p.layers[1].mask.copy()
    removed=p.subtract_layer(0,1)
    expected=np.zeros((7,7),bool);expected[3,2:4]=True
    assert removed==1
    assert np.array_equal(p.output_mask(0),expected)
    assert np.array_equal(p.layers[0].mask,target_before)
    assert np.array_equal(p.layers[1].mask,source_before)


def test_output_space_subtractions_round_trip(tmp_path):
    p=project();p.layers[0].mask[:]=True;p.layers[1].mask[:]=False;p.layers[1].mask[0,0]=True
    p.subtract_layer(0,1)
    path=tmp_path/"subtraction.hbox";p.save(path);loaded=Project.load(path)
    assert np.array_equal(loaded.output_mask(0),p.output_mask(0))
    assert np.array_equal(loaded.layers[0].output_subtractions,p.layers[0].output_subtractions)


def test_subtract_layer_rejects_subtracting_itself():
    p=project()
    with np.testing.assert_raises_regex(ValueError,"cannot be subtracted from itself"):
        p.subtract_layer(1,1)


def test_add_layer_unions_source_mask_into_target_only():
    p=project()
    p.layers[0].mask[:]=False;p.layers[0].mask[0:2,0:3]=True
    p.layers[1].mask[:]=False;p.layers[1].mask[1:,1:]=True
    source_before=p.layers[0].mask.copy()
    added=p.add_layer(1,0)
    expected=np.array([[1,1,1,0],[1,1,1,1],[0,1,1,1]],bool)
    assert added==4
    assert np.array_equal(p.layers[1].mask,expected)
    assert np.array_equal(p.layers[0].mask,source_before)


def test_add_layer_rejects_adding_itself():
    p=project()
    with np.testing.assert_raises_regex(ValueError,"cannot be added to itself"):
        p.add_layer(1,1)


def test_fill_uncovered_replaces_target_with_gaps_between_previous_layers():
    p=project()
    p.layers[0].mask[:]=False;p.layers[0].mask[0,0:2]=True
    p.layers[1].mask[:]=False;p.layers[1].mask[1,1:3]=True
    p.layers[2].mask[:]=False;p.layers[2].mask[2,3]=True
    following_before=p.layers[2].mask.copy()
    covered=p.fill_uncovered(1)
    expected=~p.output_mask(0)
    assert covered==int(expected.sum())
    assert np.array_equal(p.output_mask(1),expected)
    assert not p.layers[1].mask.any()
    assert np.array_equal(p.layers[2].mask,following_before)


def test_fill_uncovered_expands_gap_by_requested_pixel_overlap():
    image=np.zeros((7,7,3),np.uint8)
    previous=np.ones((7,7),bool);previous[3,3]=False
    target=np.zeros((7,7),bool);following=np.zeros((7,7),bool);following[0,0]=True
    p=Project(image,[Layer("previous",previous),Layer("target",target),Layer("following",following.copy())])
    covered=p.fill_uncovered(1,2)
    yy,xx=np.indices((7,7));expected=(yy-3)**2+(xx-3)**2<=4
    assert covered==int(expected.sum())
    assert np.array_equal(p.output_mask(1),expected)
    assert np.array_equal(p.layers[2].mask,following)


def test_fill_uncovered_treats_previous_pen_strokes_as_covered():
    p=project()
    p.layers[0].mask[:]=False
    p.layers[0].mask[0,0]=True
    p.layers[0].output_additions[:]=False
    p.layers[0].output_additions[1,2]=True
    covered=p.fill_uncovered(1)
    expected=np.ones(p.image.shape[:2],bool)
    expected[0,0]=False
    expected[1,2]=False
    assert covered==int(expected.sum())
    assert np.array_equal(p.output_mask(1),expected)


def test_fill_uncovered_uses_previous_layers_after_output_transform():
    image=np.zeros((7,7,3),np.uint8)
    previous=np.zeros((7,7),bool);previous[3,1]=True
    p=Project(image,[Layer("previous",previous,offset_x=3),Layer("target",np.zeros((7,7),bool))])
    covered=p.fill_uncovered(1)
    expected=np.ones((7,7),bool);expected[3,4]=False
    assert covered==48
    assert np.array_equal(p.output_mask(1),expected)


def test_fill_uncovered_is_exact_when_target_has_output_transform():
    image=np.zeros((7,7,3),np.uint8)
    previous=np.zeros((7,7),bool);previous[3,3]=True
    target=Layer("target",np.zeros((7,7),bool),offset_x=2,scale_x=.5,scale_y=1.5)
    p=Project(image,[Layer("previous",previous),target])
    p.fill_uncovered(1)
    expected=np.ones((7,7),bool);expected[3,3]=False
    assert np.array_equal(p.output_mask(1),expected)


def test_fill_uncovered_rejects_negative_overlap():
    with np.testing.assert_raises_regex(ValueError,"cannot be negative"):
        project().fill_uncovered(1,-1)


def test_fill_layer_mask_selects_entire_active_image_and_reports_added_pixels():
    p=project(np.zeros((3,4),bool));p.layers[1].mask[0,0]=True
    added=p.fill_layer_mask(1)
    assert added==11
    assert p.layers[1].mask.all()


def test_fill_layer_mask_rebinds_layer_to_active_source_before_selecting_all():
    p=project(np.zeros((3,4),bool))
    p.sources.append(SourceImage("second",np.zeros((3,4,3),np.uint8)))
    p.select_source(1)
    p.layers[0].source_index=0;p.layers[0].mask[0,0]=True
    assert p.fill_layer_mask(0)==12
    assert p.layers[0].source_index==1 and p.layers[0].mask.all()


def test_magic_wand_selects_only_connected_similar_pixels():
    image=np.zeros((4,6,3),np.uint8);image[:,:2]=[100,100,100];image[:,2:4]=[108,108,108];image[:,4:]=[100,100,100]
    strict=magic_wand_region(image,0,0,0);tolerant=magic_wand_region(image,0,0,10)
    assert strict[:,:2].all() and not strict[:,2:].any()
    assert tolerant.all()


def test_thickness_mapping_and_constraints():
    out=thickness_map(np.array([0,.5,1]),.8,3)
    assert np.allclose(out,[3,1.9,.8]) and out.min()>=.8 and out.max()<=3


def test_physical_dimensions_and_alignment():
    p=project(); assert p.pixel_mm==30 and p.height_mm==90
    a=voxel_surface(p.layers[0].mask,layer_thickness(p,0),p.pixel_mm)
    assert np.isclose(a[...,0].min(),0) and np.isclose(a[...,0].max(),120)
    assert np.isclose(a[...,1].max(),90)


def test_transmission_equation():
    p=project(); actual=combined_transmission(p)
    expected=np.exp(-sum(l.material.attenuation_per_mm*layer_thickness(p,i) for i,l in enumerate(p.layers)))
    assert np.allclose(actual,expected)


def test_rear_cutouts_increase_transmission():
    p=project(); before=combined_transmission(p); changed=optimize_rear_openings(p,0); after=combined_transmission(p)
    assert changed==24 and np.all(after>before) and not p.layers[1].mask.any() and not p.layers[2].mask.any()


def test_rear_cutouts_respect_fixed_nozzle_diameter():
    image=np.full((9,9,3),128,np.uint8)
    foreground=np.zeros((9,9),bool);foreground[4,4]=True
    rear=np.ones((9,9),bool)
    p=Project(image,[Layer("front",foreground,LayerMode.LITHOPHANE),
                     Layer("rear",rear,LayerMode.SILHOUETTE)],width_mm=.9)
    optimize_rear_openings(p)
    opening=~p.layers[1].mask
    ys,xs=np.where(opening)
    assert (xs.max()-xs.min()+1)*p.pixel_mm >= OPTIMIZER_NOZZLE_DIAMETER_MM
    assert (ys.max()-ys.min()+1)*p.pixel_mm >= OPTIMIZER_NOZZLE_DIAMETER_MM


def test_backlit_and_layer_depth_use_nozzle_resolved_detail():
    image=np.zeros((8,8,3),np.uint8);image[:,4]=255
    mask=np.ones((8,8),bool)
    p=Project(image,[Layer("front",mask,LayerMode.LITHOPHANE,Material(.8,3,.7))],width_mm=.8)
    raw=luminance(p.adjusted_image())
    resolved=nozzle_resolved_values(raw,p.pixel_mm)
    heights=layer_thickness(p,0)
    assert not np.array_equal(raw,resolved)
    assert np.allclose(heights,thickness_map(resolved,.8,3))
    expected=np.clip(np.exp(-.7*heights)*255,0,255).astype(np.uint8)
    assert np.array_equal(preview_u8(p),expected)


def test_components():
    mask=np.array([[1,0,1],[1,0,0]],bool)
    assert sorted(map(len,connected_components(mask)))==[1,2]


def test_small_sparse_islands_and_pinholes_are_cleaned_by_physical_area():
    mask=np.zeros((12,12),bool);mask[2:10,2:10]=True
    mask[5,5]=False;mask[0,0]=True
    cleaned=remove_small_features(mask,pixel_mm=.25,minimum_area_mm2=.2)
    assert not cleaned[0,0]
    assert cleaned[5,5]
    assert cleaned[2:10,2:10].all()


def test_hole_cleanup_can_be_more_aggressive_than_island_cleanup():
    mask=np.zeros((20,20),bool);mask[2:18,2:18]=True
    mask[8:12,8:12]=False;mask[0:2,0:2]=True
    cleaned=remove_small_features(mask,.25,minimum_area_mm2=.2,maximum_hole_area_mm2=2)
    assert cleaned[0:2,0:2].all()
    assert cleaned[8:12,8:12].all()


def test_minimum_width_filter_removes_long_thin_regions_but_keeps_wide_regions():
    from lightbox.geometry import remove_thin_features
    mask=np.zeros((20,24),bool)
    mask[2:18,2]=True                 # Large area, but only 0.1 mm wide.
    mask[5:15,10:20]=True             # Wide enough for a 0.4 mm line.
    cleaned=remove_thin_features(mask,pixel_mm=.1,minimum_width_mm=.4)
    assert not cleaned[:,2].any()
    assert cleaned[7:13,12:18].all()
    assert np.array_equal(remove_thin_features(mask,.1,0),mask)


def test_minimum_width_filter_rejects_invalid_physical_values():
    from lightbox.geometry import remove_thin_features
    mask=np.ones((3,3),bool)
    with np.testing.assert_raises(ValueError):remove_thin_features(mask,0,.4)
    with np.testing.assert_raises(ValueError):remove_thin_features(mask,.1,-.1)


def test_minimum_width_filter_is_applied_to_every_mask():
    from lightbox.geometry import remove_thin_features_from_masks
    thin=np.zeros((12,12),bool);thin[1:11,2]=True
    wide=np.zeros((12,12),bool);wide[2:10,5:11]=True
    cleaned=remove_thin_features_from_masks([thin,wide],.1,.4)
    assert len(cleaned)==2
    assert not cleaned[0].any()
    assert cleaned[1][4:8,7:9].all()


def test_project_width_filter_cleans_every_layer_and_segment_mask():
    thin=np.zeros((12,12),bool);thin[1:11,2]=True
    wide=np.zeros((12,12),bool);wide[2:10,5:11]=True
    image=np.zeros((12,12,3),np.uint8)
    p=Project(image,[Layer("thin",thin.copy()),Layer("wide",wide.copy())],width_mm=1.2)
    p.segment_masks=[thin.copy(),wide.copy()]
    removed=p.remove_thin_features_from_all_masks(.4)
    assert removed>0
    assert not p.layers[0].mask.any() and not p.segment_masks[0].any()
    assert p.layers[1].mask[4:8,7:9].all()
    assert p.segment_masks[1][4:8,7:9].all()


def test_each_source_keeps_an_independent_segmentation_cache():
    first=np.full((5,6,3),40,np.uint8);second=np.full((5,6,3),210,np.uint8)
    first_segment=np.zeros((5,6),bool);first_segment[1,1]=True
    second_segment=np.zeros((5,6),bool);second_segment[3,4]=True
    p=Project(first,[Layer("layer",np.ones((5,6),bool))],segment_masks=[first_segment],
              sources=[SourceImage("first.png",first,[first_segment]),SourceImage("second.png",second,[second_segment])])
    assert p.segment_masks[0][1,1]
    p.select_source(1)
    assert np.array_equal(p.image,second) and p.segment_masks[0][3,4]
    p.segment_masks[0][0,0]=True
    p.select_source(0)
    assert p.segment_masks[0][1,1] and not p.segment_masks[0][0,0]
    p.select_source(1)
    assert p.segment_masks[0][0,0]


def test_rebinding_layer_to_current_image_clears_previous_image_mask():
    first=np.zeros((5,6,3),np.uint8);second=np.full((5,6,3),200,np.uint8)
    old_mask=np.ones((5,6),bool)
    p=Project(first,[Layer("current",old_mask.copy(),source_index=0)],
              sources=[SourceImage("first",first),SourceImage("second",second)])
    changed=p.prepare_layer_for_source(0,1)
    assert changed and p.layers[0].source_index==1
    assert not p.layers[0].mask.any()
    assert np.array_equal(old_mask,np.ones((5,6),bool))
    p.layers[0].mask[2,3]=True
    assert not p.sources[0].image[2,3].any()


def test_preparing_layer_for_same_image_preserves_current_mask():
    p=project();before=p.layers[0].mask.copy()
    assert p.prepare_layer_for_source(0,0) is False
    assert np.array_equal(p.layers[0].mask,before)


def test_removing_source_clears_its_layers_and_reindexes_other_layers():
    shape=(5,6);images=[np.full((*shape,3),value,np.uint8) for value in (20,100,220)]
    layers=[Layer("removed",np.ones(shape,bool),source_index=1),Layer("shifted",np.ones(shape,bool),source_index=2)]
    p=Project(images[0],layers,sources=[SourceImage("one",images[0]),SourceImage("two",images[1]),SourceImage("three",images[2])],active_source_index=1)
    cleared=p.remove_source(1)
    assert cleared==[0] and len(p.sources)==2
    assert p.layers[0].source_index==1 and not p.layers[0].mask.any()
    assert p.layers[1].source_index==1 and p.layers[1].mask.all()
    assert p.active_source.name=="three"


def test_project_refuses_to_remove_its_only_source():
    p=project()
    with np.testing.assert_raises_regex(ValueError,"at least one"):
        p.remove_source(0)


def test_layer_transform_moves_output_without_changing_source_mask():
    image=np.zeros((7,7,3),np.uint8);mask=np.zeros((7,7),bool);mask[3,3]=True
    layer=Layer("moved",mask.copy(),offset_x=2,offset_y=-1,scale=1)
    p=Project(image,[layer])
    rendered=p.output_mask(0)
    assert rendered[2,5] and rendered.sum()==1
    assert np.array_equal(layer.mask,mask)


def test_layer_can_resize_width_and_height_independently():
    image=np.zeros((9,9,3),np.uint8);mask=np.zeros((9,9),bool);mask[3:6,3:6]=True
    layer=Layer("resized",mask.copy(),scale_x=2,scale_y=1/3)
    p=Project(image,[layer])
    rendered=p.output_mask(0)
    ys,xs=np.where(rendered)
    assert xs.max()-xs.min()+1==6
    assert ys.max()-ys.min()+1==1
    assert np.array_equal(layer.mask,mask)


def test_layer_can_rotate_output_without_changing_source_mask():
    image=np.zeros((7,7,3),np.uint8);mask=np.zeros((7,7),bool);mask[2,3:6]=True
    layer=Layer("rotated",mask.copy(),rotation_degrees=90)
    rendered=Project(image,[layer]).output_mask(0)
    assert rendered[1:4,2].all() and rendered.sum()==3
    assert np.array_equal(layer.mask,mask)


def test_mouse_transform_deltas_move_resize_and_rotate_without_editing_mask():
    image=np.zeros((10,12,3),np.uint8);mask=np.zeros((10,12),bool);mask[3:7,4:8]=True
    p=Project(image,[Layer("piece",mask.copy())])
    p.transform_layer(0,"move",3,-2)
    assert (p.layers[0].offset_x,p.layers[0].offset_y)==(3,-2)
    p.transform_layer(0,"resize",2,4)
    assert (p.layers[0].output_scale_x,p.layers[0].output_scale_y)==(1.5,2)
    p.transform_layer(0,"rotate",45,0)
    p.transform_layer(0,"rotate",150,0)
    assert p.layers[0].rotation_degrees==-165
    assert np.array_equal(p.layers[0].mask,mask)


def test_output_pen_additions_are_independent_of_source_mask_and_transform():
    image=np.zeros((8,10,3),np.uint8);mask=np.zeros((8,10),bool);mask[2,2]=True
    additions=np.zeros((8,10),bool);additions[6,8]=True
    layer=Layer("piece",mask.copy(),offset_x=2,scale_x=2,output_additions=additions.copy())
    p=Project(image,[layer]);output=p.output_mask(0)
    assert output[2,1] and output[6,8]
    p.transform_layer(0,"move",2,0)
    moved=p.output_mask(0)
    assert moved[2,3] and moved[6,8]
    assert np.array_equal(layer.mask,mask) and np.array_equal(layer.output_additions,additions)


def test_output_pen_additions_round_trip(tmp_path):
    p=project();addition=np.zeros(p.image.shape[:2],bool);addition[0,1]=True;p.layers[0].output_additions=addition
    path=tmp_path/"pen.hbox";p.save(path);loaded=Project.load(path)
    assert np.array_equal(loaded.layers[0].output_additions,addition)


def test_multi_image_project_round_trip_preserves_sources_caches_and_transforms(tmp_path):
    first=np.zeros((4,5,3),np.uint8);second=np.full((4,5,3),200,np.uint8)
    segment=np.zeros((4,5),bool);segment[1:3,2]=True
    layer=Layer("from second",segment.copy(),source_index=1,offset_x=1,offset_y=2,scale=.75,scale_x=.5,scale_y=1.25,rotation_degrees=32)
    p=Project(first,[layer],sources=[SourceImage("one.png",first),SourceImage("two.png",second,[segment.copy()])],active_source_index=1)
    path=tmp_path/"multi.hbox";p.save(path);loaded=Project.load(path)
    assert [source.name for source in loaded.sources]==["one.png","two.png"]
    assert np.array_equal(loaded.sources[1].segment_masks[0],segment)
    restored=loaded.layers[0]
    assert (restored.source_index,restored.offset_x,restored.offset_y,restored.scale)==(1,1,2,.75)
    assert (restored.output_scale_x,restored.output_scale_y)==(.5,1.25)
    assert restored.rotation_degrees==32


def test_adjacent_links_form_physical_layer_groups():
    p=project()
    assert p.physical_layer_groups()==[[0],[1],[2]]
    assert p.toggle_layer_link(0) is True
    assert p.physical_layer_groups()==[[0,1],[2]]
    assert p.toggle_layer_link(1) is True
    assert p.physical_layer_groups()==[[0,1,2]]
    assert p.toggle_layer_link(0) is False
    assert p.physical_layer_groups()==[[0],[1,2]]


def test_linked_layers_keep_independent_transforms_but_share_physical_mask():
    image=np.zeros((7,9,3),np.uint8)
    left=np.zeros((7,9),bool);left[3,2]=True
    right=np.zeros((7,9),bool);right[3,6]=True
    p=Project(image,[Layer("left",left,offset_x=1,fuse_with_next=True),Layer("right",right,offset_x=-1)])
    combined=p.physical_mask(p.physical_layer_groups()[0])
    assert combined[3,3] and combined[3,5] and combined.sum()==2
    assert left[3,2] and right[3,6]


def test_linked_layers_export_as_one_physical_stl(tmp_path):
    p=project();p.layers[0].fuse_with_next=True
    paths=export(p,tmp_path)
    assert [path.name for path in paths]==["layer_1.stl","layer_2.stl"]


def test_layer_links_round_trip_in_project_file(tmp_path):
    p=project();p.layers[0].fuse_with_next=True
    path=tmp_path/"linked.hbox";p.save(path);loaded=Project.load(path)
    assert loaded.layers[0].fuse_with_next is True
    assert loaded.physical_layer_groups()==[[0,1],[2]]


def test_frame_and_bridges_connect_islands():
    mask=np.zeros((8,10),bool); mask[3:5,5:7]=True
    assert len(connected_components(add_frame_and_bridges(mask)))==1


def test_hidden_bridge_cleanup_preserves_diagonal_cutouts_away_from_paths():
    mask=np.zeros((12,12),bool);mask[4:,:]=True;mask[1:3,8:10]=True
    mask[7,3]=False;mask[8,4]=False
    result=add_hidden_bridges(mask,np.zeros_like(mask),1,frame_width_mm=1)
    assert not result[7,3]
    assert not result[8,4]


def test_hidden_bridges_do_not_fill_islands_inside_enclosed_cutouts():
    mask=np.ones((15,15),bool);mask[4:11,4:11]=False;mask[6:9,6:9]=True
    result=add_hidden_bridges(mask,np.zeros_like(mask),1,bridge_width_mm=3,frame_width_mm=1)
    assert np.array_equal(result[4:11,4:11],mask[4:11,4:11])


def test_mesh_is_watertight_and_reimports(tmp_path):
    mask=np.array([[1,1],[1,0]],bool); heights=np.array([[1.,2.],[1.5,0]],np.float32)
    triangles=voxel_surface(mask,heights,1.0); assert is_watertight(triangles)
    path=tmp_path/"layer.stl"; write_ascii_stl(path,triangles)
    text=path.read_text(); assert text.startswith("solid layer") and text.count("facet normal")==len(triangles)
    binary=tmp_path/"layer-binary.stl";write_binary_stl(binary,triangles)
    assert binary.stat().st_size==84+50*len(triangles)


def test_project_roundtrip(tmp_path):
    p=project();p.print_smoothing_mm=.7; path=tmp_path/"x.hbox"; p.save(path); q=Project.load(path)
    assert q.width_mm==p.width_mm and np.array_equal(q.image,p.image)
    assert q.print_smoothing_mm==.7
    assert all(np.array_equal(a.mask,b.mask) for a,b in zip(p.layers,q.layers))


def test_stl_export_saves_reloadable_project_without_3mf(tmp_path):
    p=project(); paths=export(p,tmp_path)
    saved=Project.load(tmp_path/"project.hbox")
    assert len(paths)==len(p.layers)
    assert saved.width_mm==p.width_mm
    assert all(np.array_equal(a.mask,b.mask) for a,b in zip(p.layers,saved.layers))
    assert not (tmp_path/"model.3mf").exists()


def test_stl_export_does_not_require_watertight_meshes(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,"is_watertight",lambda triangles: False)
    paths=cli.export(project(),tmp_path)
    assert len(paths)==3
    assert all(path.exists() for path in paths)


def test_export_dimensions_resize_every_stl_to_requested_size(tmp_path,monkeypatch):
    written=[]
    monkeypatch.setattr(cli,"write_binary_stl",lambda path,triangles,name: (written.append(triangles.copy()),path.touch()))
    paths=export(project(),tmp_path,dimensions_mm=(109,144,2.65))
    assert len(paths)==len(written)==3
    for triangles in written:
        extents=triangles.max(axis=(0,1))-triangles.min(axis=(0,1))
        assert np.allclose(extents,[109,144,2.65],atol=1e-4)


def test_export_dimension_prompt_defaults():
    assert DEFAULT_STL_DIMENSIONS_MM==(109.0,144.0,2.65)


def test_export_dimensions_must_be_positive():
    triangles=np.array([[[0,0,0],[1,0,0],[0,1,1]]],np.float32)
    import pytest
    with pytest.raises(ValueError,match="must be positive"):
        scale_mesh_to_dimensions(triangles,109,144,0)


def test_project_crop_keeps_active_source_assets_aligned_on_same_canvas():
    p=project();p.image=np.arange(3*4*3,dtype=np.uint8).reshape(3,4,3)
    p.depth_map=np.arange(12,dtype=np.float32).reshape(3,4)
    segment=np.zeros((3,4),bool);segment[1:,1:]=True;p.segment_masks=[segment]
    original_width=p.width_mm;p.crop(1,1,4,3)
    assert p.image.shape==(3,4,3) and p.depth_map.shape==(3,4)
    assert all(layer.mask.shape==(3,4) for layer in p.layers)
    assert p.segment_masks[0].shape==(3,4)
    assert p.width_mm==original_width


def test_crop_changes_only_selected_source_and_its_layers():
    first=np.zeros((6,8,3),np.uint8);first[1:5,2:6]=100
    second=np.full((6,8,3),220,np.uint8)
    first_mask=np.zeros((6,8),bool);first_mask[1:5,2:6]=True
    second_mask=np.zeros((6,8),bool);second_mask[0,0]=True
    p=Project(first,[Layer("first",first_mask.copy(),source_index=0),Layer("second",second_mask.copy(),source_index=1)],
              sources=[SourceImage("first",first),SourceImage("second",second)])
    second_before=p.sources[1].image.copy();second_mask_before=p.layers[1].mask.copy()
    p.crop(2,1,6,5)
    assert p.sources[0].image.shape==first.shape
    assert not np.array_equal(p.sources[0].image,first)
    assert np.array_equal(p.sources[1].image,second_before)
    assert np.array_equal(p.layers[1].mask,second_mask_before)


def test_crop_composes_with_existing_layer_transform():
    image=np.zeros((10,10,3),np.uint8)
    mask=np.zeros((10,10),bool);mask[2:8,2:8]=True
    layer=Layer("transformed",mask,offset_x=3,offset_y=-2,scale_x=1.5,scale_y=.8)
    p=Project(image,[layer])
    p.crop(2,2,8,8)
    assert np.isclose(layer.scale_x,.9)
    assert np.isclose(layer.scale_y,.48)
    assert np.isclose(layer.offset_x,3)
    assert np.isclose(layer.offset_y,-2)


def test_off_center_crop_preserves_final_canvas_position_after_transform():
    image=np.zeros((10,10,3),np.uint8)
    mask=np.zeros((10,10),bool);mask[2:6,4:8]=True
    layer=Layer("transformed",mask,offset_x=1,offset_y=2,scale_x=2,scale_y=1.5)
    p=Project(image,[layer])
    p.crop(4,2,8,6)
    assert np.isclose(layer.scale_x,.8) and np.isclose(layer.scale_y,.6)
    assert np.isclose(layer.offset_x,3) and np.isclose(layer.offset_y,.5)


def test_cli_folder_discovery_and_report(tmp_path):
    source=tmp_path/"inputs"; source.mkdir(); Image.fromarray(np.full((6,8,3),128,np.uint8)).save(source/"one.png")
    (source/"ignore.txt").write_text("not an image")
    assert discover_images(source)==[source/"one.png"]
    report=generate_one(source/"one.png",tmp_path/"output",80,True)
    assert report["valid"] and len(report["exports"])==3
    assert all(path.endswith(".stl") for path in report["exports"])
    assert not (tmp_path/"output"/"model.3mf").exists()
    assert (tmp_path/"output"/"report.json").exists()


def test_ai_segment_assignment_preserves_whole_objects():
    depth=np.tile(np.linspace(0,1,6,dtype=np.float32),(4,1)); segment=np.zeros((4,6),bool);segment[:,2:4]=True
    layers=assign_depth_layers(depth,[segment],3)
    assert len(layers)==3 and np.all(sum(x.astype(np.uint8) for x in layers)==1)
    memberships=[bool(np.any(x&segment)) for x in layers]
    assert sum(memberships)==1


def test_ai_assignment_uses_adaptive_depth_planes():
    # All values occupy the near half of the normalized range. Fixed-width
    # bins would leave the rear layer empty; adaptive planes still separate it.
    depth=np.tile(np.linspace(.55,1,30,dtype=np.float32),(6,1))
    layers=assign_depth_layers(depth,None,3)
    assert all(mask.any() for mask in layers)
    assert np.all(sum(mask.astype(np.uint8) for mask in layers)==1)


def test_ai_segment_depth_ignores_noisy_boundary():
    depth=np.full((9,9),.1,np.float32);segment=np.zeros_like(depth,bool);segment[1:8,1:8]=True
    depth[segment]=.9
    # Simulate a depth-estimation halo around the object's boundary.
    depth[1,1:8]=depth[7,1:8]=.05;depth[1:8,1]=depth[1:8,7]=.05
    layers=assign_depth_layers(depth,[segment],3)
    assert np.all(layers[0][segment])


def test_ai_assignment_is_independent_of_segment_order():
    depth=np.tile(np.linspace(.05,.95,12,dtype=np.float32),(8,1))
    left=np.zeros_like(depth,bool);left[1:7,1:7]=True
    right=np.zeros_like(depth,bool);right[2:6,5:11]=True
    forward=assign_depth_layers(depth,[left,right],3)
    reverse=assign_depth_layers(depth,[right,left],3)
    assert all(np.array_equal(a,b) for a,b in zip(forward,reverse))


def test_ai_depth_persists_in_project(tmp_path):
    p=project(); depth=normalize_depth(np.arange(12,dtype=np.float32).reshape(3,4),(3,4)); masks=assign_depth_layers(depth,None,3)
    apply_ai_decomposition(p,depth,masks); path=tmp_path/"ai.hbox";p.save(path);q=Project.load(path)
    assert np.allclose(q.depth_map,depth)


def test_local_ai_uses_depth_anything_v2_for_layer_depth(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    class Prediction:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.array([[1.,2.],[3.,4.]],np.float32)
    def pipeline(task,model,device):
        calls.append((task,model,device))
        return lambda image:{"predicted_depth":Prediction()}
    ai=LocalAI(use_segmentation=False)
    torch=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)))
    monkeypatch.setattr(ai,"_imports",lambda:(torch,pipeline))
    messages=[]
    depth,segments=ai.segment_image(np.zeros((2,2,3),np.uint8),messages.append)
    assert calls==[("depth-estimation","depth-anything/Depth-Anything-V2-Small-hf",-1)]
    assert depth.shape==(2,2) and 0<=depth.min()<=depth.max()<=1
    assert segments==[]
    assert messages[-2:]==["Estimating relative depth…","Found 0 editable segments…"]


def test_editable_segments_persist_in_project(tmp_path):
    p=project();segment=np.zeros(p.image.shape[:2],bool);segment[1:,1:3]=True
    p.segment_masks=[segment];path=tmp_path/"segments.hbox";p.save(path);q=Project.load(path)
    assert len(q.segment_masks)==1 and np.array_equal(q.segment_masks[0],segment)


def test_prompted_objects_map_to_distinct_layers():
    depth=np.full((5,6),.5,np.float32); left=np.zeros_like(depth,bool);right=np.zeros_like(depth,bool)
    left[1:4,0:2]=True;right[1:4,4:6]=True;layers=assign_prompted_layers(depth,[left,right],3)
    assert np.all(layers[0][left]) and np.all(layers[1][right]) and not np.any(layers[0]&right)
    assert parse_points("0.2,0.3;0.8,0.7")==[(.2,.3),(.8,.7)]


def test_segmentation_quality_selects_checkpoint():
    assert LocalAI(segmentation_quality="accurate").segmentation_model==SAM_MODELS["accurate"]
    assert LocalAI(segmentation_quality="base").segmentation_model==SAM_MODELS["base"]
    assert LocalAI(segmentation_quality="balanced").segmentation_model==SAM_MODELS["balanced"]
    assert LocalAI(segmentation_quality="fast").segmentation_model==SAM_MODELS["fast"]
    assert LocalAI(segmentation_quality="hq").segmentation_model==SAM_MODELS["accurate"]


def test_minimum_segment_area_is_validated():
    assert LocalAI(minimum_segment_area=100).minimum_segment_area==100
    import pytest
    with pytest.raises(ValueError,match="cannot be negative"):
        LocalAI(minimum_segment_area=-1)


def test_small_segments_are_filtered_by_pixel_area():
    small=np.zeros((10,10),bool);small[:2,:2]=True
    large=np.zeros((10,10),bool);large[2:7,2:7]=True
    filtered=filter_segments([small,large],(10,10),5)
    assert len(filtered)==1 and np.array_equal(filtered[0],large)
    assert len(filter_segments([small,large],(10,10),0))==2


def test_project_supports_more_than_five_layers(tmp_path):
    source=tmp_path/"many.png";Image.fromarray(np.full((8,10,3),100,np.uint8)).save(source)
    p=Project.from_image(source,layer_count=7)
    assert len(p.layers)==7 and p.layers[-1].mask.all()
    p.add_blank_layer()
    assert len(p.layers)==8 and not p.layers[-1].mask.any()
    p.set_layer_count(6);assert len(p.layers)==6 and [x.name for x in p.layers]==[f"Layer {i}" for i in range(1,7)]


def test_new_layers_inherit_shared_layer_settings(tmp_path):
    source=tmp_path/"layers.png";Image.fromarray(np.full((8,10,3),100,np.uint8)).save(source)
    p=Project.from_image(source,layer_count=2)
    p.layers[0].mode=LayerMode.LITHOPHANE
    p.layers[0].material=Material(.6,2.4,1.1,.8,True)
    p.set_layer_count(4)
    assert all(layer.mode==LayerMode.LITHOPHANE for layer in p.layers[2:])
    assert all(layer.material==p.layers[0].material for layer in p.layers[2:])
    assert all(layer.material is not p.layers[0].material for layer in p.layers[2:])


def test_blank_layers_can_be_added_and_removed_without_changing_existing_masks():
    p=project();original=[layer.mask.copy() for layer in p.layers]
    p.add_blank_layer()
    assert len(p.layers)==4 and not p.layers[-1].mask.any()
    assert all(np.array_equal(mask,layer.mask) for mask,layer in zip(original,p.layers))
    p.remove_layer(1)
    assert len(p.layers)==3 and [layer.name for layer in p.layers]==["Layer 1","Layer 2","Layer 3"]
    assert np.array_equal(p.layers[0].mask,original[0]) and np.array_equal(p.layers[1].mask,original[2])


def test_layers_can_be_reordered_and_are_renumbered():
    p=project()
    p.layers[0].mask[0,0]=False;p.layers[1].mask[0,1]=False;p.layers[2].mask[0,2]=False
    identities=[id(layer) for layer in p.layers]
    p.move_layer(0,2)
    assert [id(layer) for layer in p.layers]==[identities[1],identities[2],identities[0]]
    assert [layer.name for layer in p.layers]==["Layer 1","Layer 2","Layer 3"]
    assert not p.layers[0].mask[0,1] and not p.layers[1].mask[0,2] and not p.layers[2].mask[0,0]


def test_high_resolution_image_is_decoupled_from_mesh_grid(tmp_path):
    source=tmp_path/"large.png";Image.fromarray(np.full((900,1200,3),100,np.uint8)).save(source)
    p=Project.from_image(source);assert p.image.shape[:2]==(576,768)
    p.width_mm=120;p.mesh_resolution_mm=.4;mask=np.ones(p.image.shape[:2],bool);heights=np.ones(mask.shape,np.float32)
    mesh_mask,mesh_heights,pixel_mm=mesh_inputs(p,mask,heights)
    assert mesh_mask.shape==(225,300) and mesh_heights.shape==mesh_mask.shape and np.isclose(pixel_mm,.4)


def test_image_can_be_imported_at_original_resolution(tmp_path):
    source=tmp_path/"original.png";Image.fromarray(np.full((900,1200,3),100,np.uint8)).save(source)
    p=Project.from_image(source,size=None)
    assert p.image.shape[:2]==(900,1200)


def test_stl_sampling_can_be_finer_than_nozzle_for_smooth_geometry():
    p=project();p.width_mm=4;p.mesh_resolution_mm=.15
    mask=np.ones((100,100),bool);heights=np.ones(mask.shape,np.float32)
    mesh_mask,mesh_heights,pixel_mm=mesh_inputs(p,mask,heights)
    assert mesh_mask.shape==(27,27) and mesh_heights.shape==(27,27)
    assert pixel_mm < OPTIMIZER_NOZZLE_DIAMETER_MM


def test_default_mesh_resolution_preserves_sub_nozzle_curve_detail():
    p=project()
    assert np.isclose(p.mesh_resolution_mm, .12)
    assert p.print_smoothing_mm == 0


def test_export_defaults_preserve_tiny_features_and_holes(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(cli,"remove_small_features",lambda mask,pixel_mm,minimum_area,maximum_hole_area: calls.append((minimum_area,maximum_hole_area)) or mask)
    export(project(),tmp_path)
    assert calls and set(calls)=={(0.0,0.0)}


def test_mesh_resampling_preserves_minimum_printable_thickness():
    p=project();p.width_mm=2;p.mesh_resolution_mm=1
    mask=np.zeros((4,4),bool);mask[1:3,1:3]=True
    heights=np.where(mask,.8,0).astype(np.float32)
    mesh_mask,mesh_heights,_=mesh_inputs(p,mask,heights)
    assert mesh_mask.any()
    assert np.all(mesh_heights[mesh_mask]>=.8)
    assert np.all(mesh_heights[~mesh_mask]==0)


def test_mesh_resampling_does_not_erase_fine_cutouts_for_small_reduction():
    p=project();p.width_mm=9;p.mesh_resolution_mm=1
    mask=np.ones((10,10),bool);mask[4,4:7]=False
    heights=np.where(mask,.8,0).astype(np.float32)
    mesh_mask,_,pixel_mm=mesh_inputs(p,mask,heights)
    assert mesh_mask.shape==(10,10)
    assert not mesh_mask[4,4:7].any()
    assert np.isclose(pixel_mm,.9)


def test_print_smoothing_removes_spikes_and_softens_height_changes():
    mask=np.zeros((15,15),bool);mask[3:12,3:12]=True;mask[7,12]=True
    heights=np.where(mask,.8,0).astype(np.float32);heights[3:12,8:12]=3.0
    smooth_mask,smooth_heights=smooth_print_data(mask,heights,.25,.5)
    assert not smooth_mask[7,12]
    assert smooth_mask[7,7] and smooth_mask[7,8]
    assert .8 < smooth_heights[7,7] < 3.0
    assert np.allclose((smooth_heights[smooth_mask] * 10) % 1,0,atol=1e-5)


def test_solid_backing_replaces_island_bridges():
    mask=np.zeros((3,4),bool);mask[0,0]=True;mask[2,3]=True;p=project(mask);p.structure_mode="backing";p.backing_thickness_mm=.6
    from lightbox.cli import inspect
    report=inspect(p)
    assert report["valid"] and report["structure_mode"]=="backing"
    assert all(layer["components_after_bridging"]==1 for layer in report["layers"])
    assert all(layer["minimum_thickness_mm"]>=.6 for layer in report["layers"])


def test_hidden_bridge_prefers_foreground_occlusion():
    mask=np.zeros((20,20),bool);mask[9:12,9:12]=True
    hidden=np.zeros_like(mask);hidden[:11,10]=True
    connected=add_hidden_bridges(mask,hidden,pixel_mm=.4,bridge_width_mm=.4,frame_width_mm=.8)
    assert len(connected_components(connected))==1
    frame=np.zeros_like(mask);frame[:2,:]=frame[-2:,:]=True;frame[:,:2]=frame[:,-2:]=True
    added=connected & ~mask & ~frame
    assert np.count_nonzero(added & hidden)>0
    assert np.count_nonzero(added & ~hidden)==0


def test_hidden_bridge_default_is_sturdy():
    mask=np.zeros((30,30),bool);mask[14:17,14:17]=True
    connected=add_hidden_bridges(mask,np.zeros_like(mask),pixel_mm=.4,frame_width_mm=.8)
    # The 3.2 mm default produces a nine-pixel-wide connector at 0.4 mm/pixel.
    bridge_row=connected[10:21,20]
    assert np.count_nonzero(bridge_row)>=9


def test_generated_connectors_have_increased_structural_depth():
    base=np.zeros((5,7),bool);base[2,4:6]=True
    structural=base.copy();structural[2,:6]=True
    heights=np.where(base,.8,0).astype(np.float32)
    reinforced=reinforce_connectors(base,structural,heights)
    assert np.all(reinforced[2,:4]==DEFAULT_CONNECTOR_DEPTH_MM)
    assert np.all(reinforced[base]==.8)


def test_connector_depth_never_reduces_existing_height():
    base=np.zeros((3,3),bool);structural=base.copy();structural[1,1]=True
    heights=np.zeros((3,3),np.float32);heights[1,1]=2.4
    assert np.isclose(reinforce_connectors(base,structural,heights)[1,1],2.4)


def test_hidden_bridge_connects_floating_piece_to_one_frame_side():
    mask=np.zeros((40,40),bool);mask[18:22,18:22]=True
    connected=add_hidden_bridges(mask,np.zeros_like(mask),pixel_mm=.4,bridge_width_mm=.4,frame_width_mm=.8)
    frame=np.zeros_like(mask);frame[:2,:]=frame[-2:,:]=True;frame[:,:2]=frame[:,-2:]=True
    added=connected & ~mask & ~frame
    reached=[added[2,:].any(),added[-3,:].any(),added[:,2].any(),added[:,-3].any()]
    assert sum(reached)==1


def test_outer_one_mm_is_three_mm_thick():
    mask=np.zeros((20,30),bool);heights=np.zeros(mask.shape,np.float32)
    mask,heights=reinforce_frame(mask,heights,pixel_mm=.25,width_mm=1,thickness_mm=3)
    assert np.all(heights[:4,:]==3) and np.all(heights[-4:,:]==3)
    assert np.all(heights[:,:4]==3) and np.all(heights[:,-4:]==3)
    assert not mask[5,5] and heights[5,5]==0
