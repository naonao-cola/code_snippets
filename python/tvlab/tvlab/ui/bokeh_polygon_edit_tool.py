'''
Copyright (C) 2023 TuringVision

bokeh bbox edit tool
'''
from PIL import Image
import numpy as np
from bokeh.core.properties import Enum, Int, String, Instance
from bokeh.core.enums import Dimensions
from bokeh.core.validation import error
from bokeh.models.glyphs import Rect
from bokeh.models import (ColumnDataSource, Column, Row, Div,
                          Dropdown, CustomJS, HoverTool)
from bokeh.models.tools import EditTool, Drag, Tap, WheelZoomTool
from bokeh.models import PolyDrawTool, PolyEditTool
from bokeh.events import Tap as TapEvent
from bokeh.plotting import figure
from bokeh.util.compiler import TypeScript
from bokeh.models.glyphs import ImageURL
from bokeh.models.renderers import GlyphRenderer, Renderer
from bokeh.models.glyphs import MultiLine, Patches, Rect, XYGlyph


__all__ = ['get_bokeh_polygon_edit_app']

JS_CODE = """
import {Keys} from "core/dom"
import {PanEvent, TapEvent, MoveEvent, KeyEvent, UIEvent} from "core/ui_events"
import {isArray} from "core/util/types"
import {MultiLine} from "models/glyphs/multi_line"
import {Patches} from "models/glyphs/patches"
import {GlyphRenderer} from "models/renderers/glyph_renderer"
import {PolyTool, PolyToolView} from "models/tools/edit/poly_tool"
import {HasXYGlyph} from "models/tools/edit/edit_tool"
import * as p from "core/properties"
import {bk_tool_icon_poly_edit} from "styles/icons"
import {CartesianFrame} from "models/canvas/cartesian_frame"
import {BoxAnnotation} from "models/annotations/box_annotation"

export interface HasPolyGlyph {
  glyph: MultiLine | Patches
}


export class TsPolyEditToolView extends PolyToolView {
  model: TsPolyEditTool
  _initialized: boolean = false

  _selected_renderer: GlyphRenderer | null
  _basepoint: [number, number] | null
  _drawing: boolean = false
  _add_drawing: boolean = false
  protected _base_point: [number, number] | null

  protected last_dx: number
  protected last_dy: number

  protected v_axis_only: boolean
  protected h_axis_only: boolean

  protected pan_info: {
    xrs: {[key: string]: {start: number, end: number}}
    yrs: {[key: string]: {start: number, end: number}}
    sdx: number
    sdy: number
  }


  _doubletap(ev: TapEvent): void {
    if (!this.model.active)
      return
    const point = this._map_drag(ev.sx, ev.sy, this.model.vertex_renderer)
    if (point == null)
      return
    const [x, y] = point

    // Perform hit testing
    const vertex_selected = this._select_event(ev, false, [this.model.vertex_renderer])
    const point_cds = this.model.vertex_renderer.data_source
    // Type once dataspecs are typed
    const point_glyph: any = this.model.vertex_renderer.glyph
    const [pxkey, pykey] = [point_glyph.x.field, point_glyph.y.field]
    if (vertex_selected.length && this._selected_renderer != null && !this._add_drawing) {
      // Insert a new point after the selected vertex and enter draw mode
      const index = point_cds.selected.indices[0]
      if (this._drawing) {
        this._drawing = false
        point_cds.selection_manager.clear()
      } else {
        point_cds.selected.indices = [index+1]
        if (pxkey) point_cds.get_array(pxkey).splice(index+1, 0, x)
        if (pykey) point_cds.get_array(pykey).splice(index+1, 0, y)
        this._drawing = true
      }
      point_cds.change.emit()
      this._emit_cds_changes(this._selected_renderer.data_source)
    } else {
      const is_not_selected = !this._selected_renderer
      if (!this._add_drawing)
        this._show_vertices(ev)

      if (!this._selected_renderer) {
        if (this._add_drawing) {
          this._add_drawing = false
          this._draw(ev, 'edit', true)
        } else if (is_not_selected){
          this._add_drawing = true
          this._draw(ev, 'new', true)
        }
      }
    }
  }

  _show_vertices(ev: UIEvent): void {
    if (!this.model.active)
      return

    const renderers = this._select_event(ev, false, this.model.renderers)
    if (!renderers.length) {
      this._set_vertices([], [])
      this._selected_renderer = null
      this._drawing = false
      return
    }

    const renderer = renderers[0]
    const glyph: any = renderer.glyph
    const cds = renderer.data_source
    const index = cds.selected.indices[0]
    const [xkey, ykey] = [glyph.xs.field, glyph.ys.field]
    let xs: number[]
    let ys: number[]
    if (xkey) {
      xs = cds.data[xkey][index]
      if (!isArray(xs))
        cds.data[xkey][index] = xs = Array.from(xs)
    } else {
      xs = glyph.xs.value
    }

    if (ykey) {
      ys = cds.data[ykey][index]
      if (!isArray(ys))
        cds.data[ykey][index] = ys = Array.from(ys)
    } else {
      ys = glyph.ys.value
    }
    this._selected_renderer = renderer
    this._set_vertices(xs, ys)
  }

  _move(ev: MoveEvent): void {
    if (this._drawing && this._selected_renderer != null) {
      const renderer = this.model.vertex_renderer
      const cds = renderer.data_source
      const glyph: any = renderer.glyph
      const point = this._map_drag(ev.sx, ev.sy, renderer)
      if (point == null)
        return
      let [x, y] = point
      const indices = cds.selected.indices
      ;[x, y] = this._snap_to_vertex(ev, x, y)
      cds.selected.indices = indices
      const [xkey, ykey] = [glyph.x.field, glyph.y.field]
      const index = indices[0]
      if (xkey) cds.data[xkey][index] = x
      if (ykey) cds.data[ykey][index] = y
      cds.change.emit()
      this._selected_renderer.data_source.change.emit()
      return
    }
    if (!this._selected_renderer) {
      if (this._add_drawing) {
        this._draw(ev, 'edit')
      }
    }
  }

  _draw(ev: UIEvent, mode: string, emit: boolean = false): void {
    const renderer = this.model.renderers[0]
    const point = this._map_drag(ev.sx, ev.sy, renderer)

    if (!this._initialized)
      this.activate() // Ensure that activate has been called

    if (point == null)
      return

    const [x, y] = this._snap_to_vertex(ev, ...point)

    const cds = renderer.data_source
    const glyph: any = renderer.glyph
    const [xkey, ykey] = [glyph.xs.field, glyph.ys.field]
    const xidx = cds.data[xkey].length-1
    let xs = cds.get_array<number[]>(xkey)[xidx]

    if (mode == 'edit' && emit == true && xs.length <= 2) {
      mode = 'add'
      this._add_drawing = true
    }

    if (mode == 'new') {
      this._pop_glyphs(cds, this.model.num_objects)
      if (xkey) cds.get_array(xkey).push([x, x])
      if (ykey) cds.get_array(ykey).push([y, y])
      cds.get_array('text').push(this.model.default_cls)
      cds.get_array('color').push('dodgerblue')
      cds.get_array('l').push(x)
      cds.get_array('t').push(y)
      this._pad_empty_columns(cds, [xkey, ykey, 'text', 'color', 'l', 't'])
      this._update_text_xy()
    } else if (mode == 'edit') {
      if (xkey) {
        const xs = cds.data[xkey][cds.data[xkey].length-1]
        xs[xs.length-1] = x
      }
      if (ykey) {
        const ys = cds.data[ykey][cds.data[ykey].length-1]
        ys[ys.length-1] = y
      }
    } else if (mode == 'add') {
      if (xkey) {
        const xidx = cds.data[xkey].length-1
        let xs = cds.get_array<number[]>(xkey)[xidx]
        const nx = xs[xs.length-1]
        xs[xs.length-1] = x
        if (!isArray(xs)) {
          xs = Array.from(xs)
          cds.data[xkey][xidx] = xs
        }
        xs.push(nx)
      }
      if (ykey) {
        const yidx = cds.data[ykey].length-1
        let ys = cds.get_array<number[]>(ykey)[yidx]
        const ny = ys[ys.length-1]
        ys[ys.length-1] = y
        if (!isArray(ys)) {
          ys = Array.from(ys)
          cds.data[ykey][yidx] = ys
        }
        ys.push(ny)
      }
      this._update_text_xy()
    }
    this._emit_cds_changes(cds, true, false, emit)
  }

  _tap(ev: TapEvent): void {
    const renderer = this.model.vertex_renderer
    const point = this._map_drag(ev.sx, ev.sy, renderer)
    if (point == null)
      return

    if (!this._selected_renderer) {
      if (this._add_drawing)
        this._draw(ev, 'add', true)
      else {
        const append = ev.shiftKey
        this._select_event(ev, append, [renderer])
        this._select_event(ev, append, this.model.renderers)
      }
      return
    }

    if (this._drawing && this._selected_renderer) {
      let [x, y] = point
      const cds = renderer.data_source
      // Type once dataspecs are typed
      const glyph: any = renderer.glyph
      const [xkey, ykey] = [glyph.x.field, glyph.y.field]
      const indices = cds.selected.indices
      ;[x, y] = this._snap_to_vertex(ev, x, y)
      const index = indices[0]
      cds.selected.indices = [index+1]
      if (xkey) {
        const xs = cds.get_array(xkey)
        const nx = xs[index]
        xs[index] = x
        xs.splice(index+1, 0, nx)
      }
      if (ykey) {
        const ys = cds.get_array(ykey)
        const ny = ys[index]
        ys[index] = y
        ys.splice(index+1, 0, ny)
      }
      cds.change.emit()
      this._emit_cds_changes(this._selected_renderer.data_source, true, false, true)
      return
    }
    const append = ev.shiftKey
    this._select_event(ev, append, [renderer])
    this._select_event(ev, append, this.model.renderers)
  }

  _remove_vertex(): void {
    if (!this._drawing || !this._selected_renderer)
      return
    const renderer = this.model.vertex_renderer
    const cds = renderer.data_source
    // Type once dataspecs are typed
    const glyph: any = renderer.glyph
    const indices = cds.selected.indices
    if (indices.length > 0) {
      const index = indices[0]
      const [xkey, ykey] = [glyph.x.field, glyph.y.field]
      if (xkey) cds.get_array(xkey).splice(index, 1)
      if (ykey) cds.get_array(ykey).splice(index, 1)
      cds.change.emit()
      this._emit_cds_changes(this._selected_renderer.data_source)
    }
    cds.selection_manager.clear()
    cds.change.emit()
  }

  _match_aspect(base_point: [number, number], curpoint: [number, number],
                frame: CartesianFrame): [[number, number], [number, number]] {
    // aspect ratio of plot frame
    const a = frame.bbox.aspect
    const hend = frame.bbox.h_range.end
    const hstart = frame.bbox.h_range.start
    const vend = frame.bbox.v_range.end
    const vstart = frame.bbox.v_range.start

    // current aspect of cursor-defined box
    let vw = Math.abs(base_point[0]-curpoint[0])
    let vh = Math.abs(base_point[1]-curpoint[1])

    const va = vh == 0 ? 0 : vw/vh
    const [xmod] = va >= a ? [1, va/a] : [a/va, 1]

    // OK the code blocks below merit some explanation. They do:
    //
    // compute left/right, pin to frame if necessary
    // compute top/bottom (based on new left/right), pin to frame if necessary
    // recompute left/right (based on top/bottom), in case top/bottom were pinned

    // base_point[0] is left
    let left: number
    let right: number
    if (base_point[0] <= curpoint[0]) {
      left = base_point[0]
      right = base_point[0] + vw * xmod
      if (right > hend)
        right = hend
    // base_point[0] is right
    } else {
      right = base_point[0]
      left = base_point[0] - vw * xmod
      if (left < hstart)
        left = hstart
    }

    vw = Math.abs(right - left)

    // base_point[1] is bottom
    let top: number
    let bottom: number
    if (base_point[1] <= curpoint[1]) {
      bottom = base_point[1]
      top = base_point[1] + vw/a
      if (top > vend)
        top = vend
    // base_point[1] is top
    } else {
      top = base_point[1]
      bottom = base_point[1] - vw/a
      if (bottom < vstart)
        bottom = vstart
    }

    vh = Math.abs(top - bottom)

    // base_point[0] is left
    if (base_point[0] <= curpoint[0])
      right = base_point[0] + a*vh
    // base_point[0] is right
    else
      left = base_point[0] - a*vh

    return [[left, right], [bottom, top]]
  }

  protected _compute_limits(curpoint: [number, number]): [[number, number], [number, number]] {
    const frame = this.plot_view.frame

    let base_point = this._base_point!

    let sx: [number, number]
    let sy: [number, number]
    [sx, sy] = this._match_aspect(base_point, curpoint, frame)

    return [sx, sy]
  }

  _update([sx0, sx1]: [number, number], [sy0, sy1]: [number, number], box_zoom=false): void {
    if (box_zoom) {
      if (Math.abs(sx1 - sx0) <= 5 || Math.abs(sy1 - sy0) <= 5)
        return

      const {xscales, yscales} = this.plot_view.frame

      const xrs: {[key: string]: {start: number, end: number}} = {}
      for (const name in xscales) {
        const scale = xscales[name]
        const [start, end] = scale.r_invert(sx0, sx1)
        xrs[name] = {start, end}
      }

      const yrs: {[key: string]: {start: number, end: number}} = {}
      for (const name in yscales) {
        const scale = yscales[name]
        const [start, end] = scale.r_invert(sy0, sy1)
        yrs[name] = {start, end}
      }

      const zoom_info = {xrs, yrs}

      this.plot_view.push_state('box_zoom', {range: zoom_info})
      this.plot_view.update_range(zoom_info)
    } else {
      const dx = sx1
      const dy = sy1

      const frame = this.plot_view.frame

      const new_dx = dx - this.last_dx
      const new_dy = dy - this.last_dy

      const hr = frame.bbox.h_range
      const sx_low  = hr.start - new_dx
      const sx_high = hr.end - new_dx

      const vr = frame.bbox.v_range
      const sy_low  = vr.start - new_dy
      const sy_high = vr.end - new_dy

      let ssx0: number
      let ssx1: number
      let sdx: number
      if (!this.v_axis_only) {
        ssx0 = sx_low
        ssx1 = sx_high
        sdx = -new_dx
      } else {
        ssx0 = hr.start
        ssx1 = hr.end
        sdx = 0
      }

      let ssy0: number
      let ssy1: number
      let sdy: number
      if (!this.h_axis_only) {
        ssy0 = sy_low
        ssy1 = sy_high
        sdy = -new_dy
      } else {
        ssy0 = vr.start
        ssy1 = vr.end
        sdy = 0
      }

      this.last_dx = dx
      this.last_dy = dy

      const {xscales, yscales} = frame

      const xrs: {[key: string]: {start: number, end: number}} = {}
      for (const name in xscales) {
        const scale = xscales[name]
        const [start, end] = scale.r_invert(ssx0, ssx1)
        xrs[name] = {start, end}
      }

      const yrs: {[key: string]: {start: number, end: number}} = {}
      for (const name in yscales) {
        const scale = yscales[name]
        const [start, end] = scale.r_invert(ssy0, ssy1)
        yrs[name] = {start, end}
      }

      this.pan_info = {xrs, yrs, sdx, sdy}
      this.plot_view.update_range(this.pan_info, true)
    }
  }

  _pan_start(ev: PanEvent): void {
    this.last_dx = 0
    this.last_dy = 0
    this.h_axis_only = false
    this.v_axis_only = false
    if (ev.shiftKey) {
      this._base_point = [ev.sx, ev.sy]
    } else {
      if (this._basepoint != null)
        return

      this._select_event(ev, true, this.model.renderers)
      this._select_event(ev, true, [this.model.vertex_renderer])
      this._basepoint = [ev.sx, ev.sy]

      var is_selected = false
      for (const renderer of this.model.renderers) {
        const point = this._map_drag(ev.sx, ev.sy, renderer)
        if (point == null)
          continue
        const cds = renderer.data_source
        if (cds.selected.indices.length > 0) {
          is_selected = true
          break
        }
      }

      console.log(`@@@@ _pan_start is_selected: '${is_selected}'`)
      if (!is_selected) {
        this._basepoint = null
        if (this.model.document != null)
          this.model.document.interactive_start(this.plot_model)
      }
    }
  }

  _drag_points(ev: UIEvent, renderers: (GlyphRenderer & HasXYGlyph)[]): void {
    if (this._basepoint == null)
      return
    const [bx, by] = this._basepoint
    var is_selected = false
    for (const renderer of renderers) {
      const basepoint = this._map_drag(bx, by, renderer)
      const point = this._map_drag(ev.sx, ev.sy, renderer)
      if (point == null || basepoint == null) {
        continue
      }
      const [x, y] = point
      const [px, py] = basepoint
      const [dx, dy] = [x-px, y-py]
      // Type once dataspecs are typed
      const glyph: any = renderer.glyph
      const cds = renderer.data_source
      const [xkey, ykey] = [glyph.x.field, glyph.y.field]
      for (const index of cds.selected.indices) {
        is_selected = true
        if (xkey) cds.data[xkey][index] += dx
        if (ykey) cds.data[ykey][index] += dy
      }
      cds.change.emit()
    }
    if (is_selected)
      this._basepoint = [ev.sx, ev.sy]
  }

  _pan(ev: PanEvent): void {
    if (ev.shiftKey) {
      const curpoint: [number, number] = [ev.sx, ev.sy]
      const [sx, sy] = this._compute_limits(curpoint)
      this.model.overlay.update({left: sx[0], right: sx[1], top: sy[0], bottom: sy[1]})
    } else {
      if (this._basepoint == null) {
        const sx: [number, number] = [0, ev.deltaX]
        const sy: [number, number] = [0, ev.deltaY]
        this._update(sx, sy)
        if (this.model.document != null)
          this.model.document.interactive_start(this.plot_model)
        return
      }
      const [bx, by] = this._basepoint
      this._drag_points(ev, [this.model.vertex_renderer])
      if (this._selected_renderer)
        this._selected_renderer.data_source.change.emit()

      const [nx, ny] = this._basepoint
      if (nx != ev.sx || ny != ev.sy) {
        // Process polygon/line dragging
        for (const renderer of this.model.renderers) {
          const basepoint = this._map_drag(bx, by, renderer)
          const point = this._map_drag(ev.sx, ev.sy, renderer)
          if (point == null || basepoint == null)
            continue

          const cds = renderer.data_source
          // Type once dataspecs are typed
          const glyph: any = renderer.glyph
          const [xkey, ykey] = [glyph.xs.field, glyph.ys.field]
          if (!xkey && !ykey)
            continue
          const [x, y] = point
          const [px, py] = basepoint
          const [dx, dy] = [x-px, y-py]
          for (const index of cds.selected.indices) {
            let length, xs, ys
            if (xkey) xs = cds.data[xkey][index]
            if (ykey) {
              ys = cds.data[ykey][index]
              length = ys.length
            } else {
              length = xs.length
            }
            cds.data['l'][index] += dx
            cds.data['t'][index] += dy
            for (let i = 0; i < length; i++) {
              if (xs) xs[i] += dx
              if (ys) ys[i] += dy
            }
          }
          this._update_text_xy()
          cds.change.emit()
        }
        this._basepoint = [ev.sx, ev.sy]
      }
    }
  }

  _pan_end(ev: PanEvent): void {
    if (ev.shiftKey) {
      const curpoint: [number, number] = [ev.sx, ev.sy]
      const [sx, sy] = this._compute_limits(curpoint)
      this._update(sx, sy, true)
      this.model.overlay.update({left: null, right: null, top: null, bottom: null})
      this._base_point = null
    } else {
      if (this._basepoint == null)
        return
      this._drag_points(ev, [this.model.vertex_renderer])
      this._update_text_xy()
      this._emit_cds_changes(this.model.vertex_renderer.data_source, false, true, true)
      if (this._selected_renderer) {
        this._emit_cds_changes(this._selected_renderer.data_source)
      } else {
        this._pan(ev)
        for (const renderer of this.model.renderers)
          this._emit_cds_changes(renderer.data_source)
      }
      this._basepoint = null
      this.h_axis_only = false
      this.v_axis_only = false
      if (this.pan_info != null)
        this.plot_view.push_state('pan', {range: this.pan_info})
    }
  }

  _keyup(ev: KeyEvent): void {
    if (!this.model.active || !this._mouse_in_frame)
      return
    let renderers: GlyphRenderer[]
    if (this._selected_renderer) {
      renderers = [this.model.vertex_renderer]
    } else {
      renderers = this.model.renderers
    }
    for (const renderer of renderers) {
      if (ev.keyCode === Keys.Backspace) {
        this._delete_selected(renderer)
        if (this._selected_renderer) {
          this._emit_cds_changes(this._selected_renderer.data_source)
        }
      } else if (ev.keyCode == Keys.Esc) {
        var is_remove = false
        if (this._add_drawing) {
          this._remove()
          this._add_drawing = false
          is_remove = true
        } if (this._drawing) {
          this._remove_vertex()
          this._drawing = false
          is_remove = true
        }
        if (is_remove) {
        } else if (this._selected_renderer) {
          this._hide_vertices()
        } else {
          this.plot_view.reset()
        }
        this._selected_renderer = null
        renderer.data_source.selection_manager.clear()
      }
    }
  }

  _update_text_xy(): void {
    const renderer = this.model.renderers[0]
    const cds = renderer.data_source
    const glyph: any = renderer.glyph
    const [xkey, ykey] = [glyph.xs.field, glyph.ys.field]
    for (let i = 0; i < cds.data[xkey].length; i++) {
      const xs = cds.data[xkey][i]
      const ys = cds.data[ykey][i]
      const ys_min = Math.min(...ys)
      var j = ys.indexOf(ys_min)
      cds.data['l'][i] = xs[j]
      cds.data['t'][i] = ys[j]
    }
  }

  _remove(): void {
    const renderer = this.model.renderers[0]
    const cds = renderer.data_source
    const glyph: any = renderer.glyph
    const [xkey, ykey] = [glyph.xs.field, glyph.ys.field]
    var need_remove = false
    if (xkey) {
      const xidx = cds.data[xkey].length-1
      const xs = cds.get_array<number[]>(xkey)[xidx]
      if (xs.length > 3) {
        xs.splice(xs.length-1, 1)
      } else {
        need_remove = true
      }
    }
    if (ykey) {
      const yidx = cds.data[ykey].length-1
      const ys = cds.get_array<number[]>(ykey)[yidx]
      if (ys.length > 3) {
        ys.splice(ys.length-1, 1)
      } else {
        need_remove = true
      }
    }
    if (need_remove) {
      const idx = cds.data[xkey].length-1
      for (const column of cds.columns()) {
        const values = cds.get_array(column)
        values.splice(idx, 1)
      }
    }
    this._update_text_xy()
    this._emit_cds_changes(cds)
  }

  activate(): void {
    if (!this.model.vertex_renderer || !this.model.active)
      return
    this._initialized = true
  }

  deactivate(): void {
    if (!this._selected_renderer) {
      return
    } else if (this._add_drawing) {
      this._remove()
      this._add_drawing = false
    } else if (this._drawing) {
      this._remove_vertex()
      this._drawing = false
    }
    this._hide_vertices()
  }
}

const DEFAULT_BOX_OVERLAY = () => {
  return new BoxAnnotation({
    level: "overlay",
    render_mode: "css",
    top_units: "screen",
    left_units: "screen",
    bottom_units: "screen",
    right_units: "screen",
    fill_color: {value: "lightgrey"},
    fill_alpha: {value: 0.5},
    line_color: {value: "black"},
    line_alpha: {value: 1.0},
    line_width: {value: 2},
    line_dash: {value: [4, 4]},
  })
}

export namespace TsPolyEditTool {
  export type Attrs = p.AttrsOf<Props>

  export type Props = PolyTool.Props & {
    overlay: p.Property<BoxAnnotation>
    drag: p.Property<boolean>
    default_cls: p.Property<string>
    num_objects: p.Property<number>
  }
}

export interface TsPolyEditTool extends TsPolyEditTool.Attrs {}

export class TsPolyEditTool extends PolyTool {
  properties: TsPolyEditTool.Props
  /*override*/ overlay: BoxAnnotation

  constructor(attrs?: Partial<TsPolyEditTool.Attrs>) {
    super(attrs)
  }

  static init_TsPolyEditTool(): void {
    this.prototype.default_view = TsPolyEditToolView

    this.define<TsPolyEditTool.Props>({
      overlay: [ p.Instance,   DEFAULT_BOX_OVERLAY ],
      drag:        [ p.Boolean, true ],
      default_cls: [ p.String, 'Other' ],
      num_objects: [ p.Int,     0    ],
    })
  }

  tool_name = "PolyEditTool\\ndouble tap: start/stop add polygon\\n"
            + "tap: add point or select polygon or show x,y,pixel value\\n"
            + "shift+tap: multi select\\n"
            + "left-dragging: move image or move point\\n"
            + "shift+left-dragging: zoom rectangular region\\n"
            + "BackSpace: delete the selected polygon\\n"
            + "Esc: reset zoom, stop add, cancel selected\\n"

  icon = bk_tool_icon_poly_edit
  event_type = ["tap" as "tap", "pan" as "pan", "move" as "move"]
  default_order = 4
}
"""


def get_bokeh_polygon_edit_app(img_path_list, source_list, labelset, desc_list, ncols=1,
                               max_size=1200, active_zoom=False, share_xy=False,
                               label_level=False):
    from bokeh.core.validation.errors import (INCOMPATIBLE_POLY_EDIT_VERTEX_RENDERER,
                                              INCOMPATIBLE_POLY_EDIT_RENDERER)
    class TsPolyEditTool(EditTool, Drag, Tap):
        __implementation__ = TypeScript(JS_CODE)
        vertex_renderer = Instance(GlyphRenderer, help="""
        The renderer used to render the vertices of a selected line or
        polygon.""")
        default_cls = String(default='Other')

        @error(INCOMPATIBLE_POLY_EDIT_VERTEX_RENDERER)
        def _check_compatible_vertex_renderer(self):
            glyph = self.vertex_renderer.glyph
            if not isinstance(glyph, XYGlyph):
                return "glyph type %s found." % type(glyph).__name__

        @error(INCOMPATIBLE_POLY_EDIT_RENDERER)
        def _check_compatible_renderers(self):
            incompatible_renderers = []
            for renderer in self.renderers:
                if not isinstance(renderer.glyph, (MultiLine, Patches)):
                    incompatible_renderers.append(renderer)
            if incompatible_renderers:
                glyph_types = ', '.join(type(renderer.glyph).__name__
                                        for renderer in incompatible_renderers)
                return "%s glyph type(s) found." % glyph_types

    img_wh_list = list()
    for img_path in img_path_list:
        im = Image.open(img_path)
        img_wh_list.append((im.width, im.height))

    def bkapp(doc):
        layout_list = list()
        first_plot = None

        for i, img_path in enumerate(img_path_list):
            img_w, img_h = img_wh_list[i]
            url = [img_path]
            source = source_list[i]
            if img_w < img_h:
                plot_h = int(min(img_h, max_size))
                plot_w = int(img_w * (plot_h / img_h))
            else:
                plot_w = int(min(img_w, max_size))
                plot_h = int(img_h * (plot_w / img_w))

            try:
                img = Image.open(img_path)
            except Exception:
                return;

            x_range = (0, img_w)
            y_range = (img_h, 0)
            if first_plot is not None and share_xy:
                x_range = first_plot.x_range
                y_range = first_plot.y_range

            plot = figure(x_range=x_range, y_range=y_range,
                          x_axis_location="above",
                          frame_width=plot_w,
                          frame_height=plot_h,
                          width_policy='min',
                          height_policy='min',
                          tools="reset")

            if first_plot is None:
                first_plot = plot

            plot.image_url(url=url, x=0, y=0, w=img_w, h=img_h, retry_attempts=3)

            t = plot.text('l', 't', 'text', text_font_size='6pt', source=source,
                          text_alpha=0.7, text_color='color',
                          selection_text_alpha=1.0,
                          selection_text_color='color',
                          nonselection_text_alpha=0.5,
                          nonselection_text_color='color')

            patches = plot.patches('x', 'y', source=source, color='color',
                                   line_alpha=0.7, fill_alpha=0.1,
                                   selection_fill_alpha=0.0,
                                   selection_line_alpha=1.0,
                                   selection_color='color',
                                   nonselection_fill_alpha=0.2,
                                   nonselection_line_alpha=0.5,
                                   nonselection_color='color')
            c = plot.circle([], [], size=5, color='red')

            menu = [(c, c+'-5' if label_level else c) for c in labelset]
            dropdown = Dropdown(label="Label:",
                                button_type="warning", menu=menu,
                                width_policy='min')
            dropdown.js_on_click(CustomJS(args=dict(source=source), code="""
            for (const index of source.selected.indices) {
              source.data['text'][index] = cb_obj.item
            }
            source.change.emit()
            source.data = source.data
            source.properties.data.change.emit()
            """))
            dropdown.disabled = True

            edit_tool = TsPolyEditTool(renderers=[patches], vertex_renderer=c, default_cls=menu[0][1])

            def_dropdown = Dropdown(label="Default: {}".format(labelset[0]),
                                    button_type="primary", menu=menu,
                                    width_policy='min')
            def_dropdown.js_on_click(CustomJS(args=dict(other=edit_tool, b=def_dropdown),
                                     code="""
                                          other.default_cls = cb_obj.item
                                          b.label = "Default: " + cb_obj.item + ""
                                          """))

            level_dropdown = None
            if label_level:
                level_dropdown = Dropdown(label="Level",
                                    button_type="danger", menu=[(str(i), str(i)) for i in range(1, 10)],
                                    width_policy='min')
                level_dropdown.js_on_click(CustomJS(args=dict(source=source), code="""
                for (const index of source.selected.indices) {
                  source.data['text'][index] = source.data['text'][index].slice(0,source.data['text'][index].length-1) + cb_obj.item
                }
                source.change.emit()
                source.data = source.data
                source.properties.data.change.emit()
                """))
                level_dropdown.disabled = True

            source.js_on_change('selected', CustomJS(args=dict(other=dropdown), code="""
            other.disabled = true
            if (this.selected.indices.length > 0) {
                other.disabled = false
            }
            """
            ))
            if label_level:
                source.js_on_change('selected', CustomJS(args=dict(other=level_dropdown), code="""
                other.disabled = true
                if (this.selected.indices.length > 0) {
                    other.disabled = false
                }
                """
                ))
            desc = ""
            if desc_list:
                desc = desc_list[i]
                desc = ("<span style=float:left;clear:left;font_size=10pt>{}</span>".format(desc))
            div = Div(width_policy='min')
            div.text = ("{}\n<span style=float:left;clear:left;font_size=10pt>".format(desc) +
                        "X: {}, Y: {}, Pixel Value:{}</span>\n".format(0, 0, 0))

            def get_tap_cb(div, desc, img, img_w, img_h):
                def display_event(ev):
                    "Build a suitable CustomJS to display the current event in the div model."
                    x = min(max(0, int(ev.x)), img_w-1)
                    y = min(max(0, int(ev.y)), img_h-1)
                    pixel = img.getpixel((x, y))
                    div.text = ("{}\n<span style=float:left;clear:left;font_size=10pt>".format(desc) +
                                "X: {}, Y: {}, Pixel Value:{}</span>\n".format(x, y, pixel))
                return display_event

            plot.on_event(TapEvent, get_tap_cb(div, desc, img, img_w, img_h))

            wheel_zoom = WheelZoomTool(dimensions='both', zoom_on_axis=False, maintain_focus=True)
            plot.add_tools(edit_tool, wheel_zoom)
            plot.toolbar.active_multi = edit_tool
            plot.toolbar.active_scroll = None
            if active_zoom:
                plot.toolbar.active_scroll = wheel_zoom
            drop_down_list = [def_dropdown, dropdown]
            if label_level:
                drop_down_list.append(level_dropdown)
            layout = Column(children=[Row(*drop_down_list), div, plot])
            layout_list.append(layout)

        num = len(layout_list)
        row_idxs = np.array_split(np.arange(num), int(np.ceil(num/ncols)))
        row_list = list()
        for idxs in row_idxs:
            row_layout = [layout_list[i] for i in idxs]
            row_list.append(Row(*row_layout))
        doc.add_root(Column(children=row_list))

    return bkapp
