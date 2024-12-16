'''
Copyright (C) 2023 TuringVision

bokeh bbox edit tool
'''
from PIL import Image
import numpy as np
from bokeh.core.properties import Enum, Int, String
from bokeh.core.enums import Dimensions
from bokeh.core.validation import error
from bokeh.models.glyphs import Rect
from bokeh.models import (ColumnDataSource, Column, Row, Div,
                          Dropdown, CustomJS, HoverTool)
from bokeh.models.tools import EditTool, Drag, Tap, WheelZoomTool
from bokeh.events import Tap as TapEvent
from bokeh.plotting import figure
from bokeh.util.compiler import TypeScript
from bokeh.models.glyphs import ImageURL


__all__ = ['get_bokeh_bbox_edit_app']

JS_CODE = """
import {Keys} from "core/dom"
import {PanEvent, TapEvent, KeyEvent, UIEvent, MoveEvent} from "core/ui_events"
import {Dimensions} from "core/enums"
import {min, max} from "core/util/array"
import * as p from "core/properties"
import {Rect} from "models/glyphs/rect"
import {XYGlyph} from "models/glyphs/xy_glyph"
import {GlyphRenderer} from "models/renderers/glyph_renderer"
import {ColumnDataSource} from "models/sources/column_data_source"
import {EditTool, EditToolView} from "models/tools/edit/edit_tool"
import {bk_tool_icon_box_edit} from "styles/icons"
import {CartesianFrame} from "models/canvas/cartesian_frame"
import {BoxAnnotation} from "models/annotations/box_annotation"

export interface HasRectCDS {
  glyph: Rect
  data_source: ColumnDataSource
}

export interface HasXYGlyph {
  glyph: XYGlyph
}

export class TsBboxEditToolView extends EditToolView {
  model: TsBboxEditTool
  _draw_basepoint: [number, number] | null
  _left_right_adj: number | 0
  _top_bottom_adj: number | 0
  _drag_boxes: [number, number, number, number] | null

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

  protected _base_point: [number, number] | null

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
    const dims = this.model.dimensions

    let base_point = this._base_point!

    let sx: [number, number]
    let sy: [number, number]
    if (this.model.match_aspect && dims == 'both')
      [sx, sy] = this._match_aspect(base_point, curpoint, frame)
    else
      [sx, sy] = this.model._get_dim_limits(base_point, curpoint, frame, dims)

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

      const dims = this.model.dimensions

      let ssx0: number
      let ssx1: number
      let sdx: number
      if ((dims == 'width' || dims == 'both') && !this.v_axis_only) {
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
      if ((dims == 'height' || dims == 'both') && !this.h_axis_only) {
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

  _tap(ev: TapEvent): void {
    if ((this._draw_basepoint != null) || (this._basepoint != null))
      return
    const append = ev.shiftKey
    this._select_event(ev, append, this.model.renderers)
  }

  _keyup(ev: KeyEvent): void {
    if (!this.model.active || !this._mouse_in_frame)
      return
    for (const renderer of this.model.renderers) {
      if (ev.keyCode === Keys.Backspace) {
        this._delete_selected(renderer)
      } else if (ev.keyCode == Keys.Esc) {
        this.plot_view.reset()
      }
    }
  }

  _set_extent([sx0, sx1]: [number, number], [sy0, sy1]: [number, number],
              append: boolean, emit: boolean = false): void {
    const renderer = this.model.renderers[0]
    const frame = this.plot_view.frame
    // Type once dataspecs are typed
    const glyph: any = renderer.glyph
    const cds = renderer.data_source
    const xscale = frame.xscales[renderer.x_range_name]
    const yscale = frame.yscales[renderer.y_range_name]
    const [x0, x1] = xscale.r_invert(sx0, sx1)
    const [y0, y1] = yscale.r_invert(sy0, sy1)
    const [x, y] = [(x0+x1)/2, (y0+y1)/2]
    const [w, h] = [Math.abs(x1-x0), Math.abs(y1-y0)]
    const [xkey, ykey] = [glyph.x.field, glyph.y.field]
    const [wkey, hkey] = [glyph.width.field, glyph.height.field]
    if (append) {
      this._pop_glyphs(cds, this.model.num_objects)
      if (xkey) cds.get_array(xkey).push(x)
      if (ykey) cds.get_array(ykey).push(y)
      if (wkey) cds.get_array(wkey).push(w)
      if (hkey) cds.get_array(hkey).push(h)
      cds.get_array('l').push(x-w/2)
      cds.get_array('t').push(y-h/2)
      cds.get_array('text').push(this.model.default_cls)
      cds.get_array('color').push('dodgerblue')
      this._pad_empty_columns(cds, [xkey, ykey, wkey, hkey, 'l', 't', 'text', 'color'])
    } else {
      const index = cds.data[xkey].length - 1
      if (xkey) cds.data[xkey][index] = x
      if (ykey) cds.data[ykey][index] = y
      if (wkey) cds.data[wkey][index] = w
      if (hkey) cds.data[hkey][index] = h
      cds.data['l'][index] = x - w/2
      cds.data['t'][index] = y - h/2
    }
    this._emit_cds_changes(cds, true, false, emit)
  }

  _get_dim_limits([sx0, sy0]: [number, number], [sx1, sy1]: [number, number],
      dims: Dimensions): [[number, number], [number, number]] {

    const frame = this.plot_view.frame
    const hr = frame.bbox.h_range
    const [hs,  he] = [min([hr.start, hr.end]), max([hr.start, hr.end])]
    let sxlim: [number, number]
    if (dims == 'width' || dims == 'both') {
      sxlim = [min([sx0, sx1]),           max([sx0, sx1])]
      sxlim = [max([sxlim[0], hs]), min([sxlim[1], he])]
    } else
      sxlim = [hs, he]

    const vr = frame.bbox.v_range
    const [vs,  ve] = [min([vr.start, vr.end]), max([vr.start, vr.end])]
    let sylim: [number, number]
    if (dims == 'height' || dims == 'both') {
      sylim = [min([sy0, sy1]),           max([sy0, sy1])]
      sylim = [max([sylim[0], vs]), min([sylim[1], ve])]
    } else
      sylim = [vs, ve]

    return [sxlim, sylim]
  }

  _update_box(ev: UIEvent, append: boolean = false, emit: boolean = false): void {
    if (this._draw_basepoint == null)
      return
    const curpoint: [number, number] = [ev.sx, ev.sy]
    const dims = this.model.dimensions
    const limits = this._get_dim_limits(this._draw_basepoint, curpoint, dims)
    if (limits != null) {
      const [sxlim, sylim] = limits
      this._set_extent(sxlim, sylim, append, emit)
    }
  }

  _doubletap(ev: TapEvent): void {
    if (!this.model.active)
      return
    if (this._draw_basepoint != null) {
      this._update_box(ev, false, true)
      this._draw_basepoint = null
    } else {
      const frame = this.plot_view.frame
      const hr = frame.bbox.h_range
      const vr = frame.bbox.v_range
      const [hs,  he] = [min([hr.start, hr.end]), max([hr.start, hr.end])]
      const [vs,  ve] = [min([vr.start, vr.end]), max([vr.start, vr.end])]
      if (ev.sy >= vs && ev.sy <= ve && ev.sx >= hs && ev.sx <= he) {
        this._draw_basepoint = [ev.sx, ev.sy]
        this._select_event(ev, true, this.model.renderers)
        this._update_box(ev, true, false)
      }
    }
  }

  _move(ev: MoveEvent): void {
    this._update_box(ev, false, false)
  }

  _pan_start(ev: PanEvent): void {
    if (ev.shiftKey) {
      this._base_point = [ev.sx, ev.sy]
    } else {
      if (this._basepoint != null)
        return
      this._select_event(ev, true, this.model.renderers)
      this._basepoint = [ev.sx, ev.sy]
      this._left_right_adj = 0
      this._top_bottom_adj = 0
      var is_selected = false
      for (const renderer of this.model.renderers) {
        const cds = renderer.data_source
        const point = this._map_drag(ev.sx, ev.sy, renderer)
        if (point == null) {
          continue
        }
        const [x, y] = point
        for (const index of cds.selected.indices) {
          var sx = cds.data['x'][index]
          var sy = cds.data['y'][index]
          var sw = cds.data['w'][index]
          var sh = cds.data['h'][index]
          var [l,t,r,b] = [sx-sw/2, sy-sh/2, sx+sw/2, sy+sh/2]
          const border_thresh_w = min([max([1, sw*0.3]), 100])
          const border_thresh_h = min([max([1, sh*0.3]), 100])

          if (Math.abs(l - x) < border_thresh_w) {
              this._left_right_adj = -1
          } else if (Math.abs(r - x) < border_thresh_w) {
              this._left_right_adj = 1
          } else {
              this._left_right_adj = 0
          }

          if (Math.abs(t - y) < border_thresh_h) {
              this._top_bottom_adj = -1
          } else if (Math.abs(b - y) < border_thresh_h) {
              this._top_bottom_adj = 1
          } else {
              this._top_bottom_adj = 0
          }
          is_selected = true
          console.log(`left_right:${this._left_right_adj}, top_bottom:${this._top_bottom_adj}`)
          break
        }
        break
      }
      if (is_selected == false) {
        this._basepoint = null
        this.last_dx = 0
        this.last_dy = 0
        const {sx, sy} = ev
        const bbox = this.plot_view.frame.bbox
        if (!bbox.contains(sx, sy)) {
          const hr = bbox.h_range
          const vr = bbox.v_range
          if (sx < hr.start || sx > hr.end)
            this.v_axis_only = true
          if (sy < vr.start || sy > vr.end)
            this.h_axis_only = true
        }

        if (this.model.document != null)
          this.model.document.interactive_start(this.plot_model)
      }
    }
  }

  _drag_points(ev: UIEvent, renderers: (GlyphRenderer & HasXYGlyph)[]): void {
    if (this._basepoint == null)
      return
    const [bx, by] = this._basepoint
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
      const [wkey, hkey] = [glyph.width.field, glyph.height.field]
      for (const index of cds.selected.indices) {
        var sx = cds.data[xkey][index]
        var sy = cds.data[ykey][index]
        var sw = cds.data[wkey][index]
        var sh = cds.data[hkey][index]
        var [l,t,r,b] = [sx-sw/2, sy-sh/2, sx+sw/2, sy+sh/2]

        if (this._left_right_adj == -1) {
            l += dx
        } else if (this._left_right_adj == 1) {
            r += dx
        } else if (this._top_bottom_adj == 0) {
            l += dx
            r += dx
        }

        if (this._top_bottom_adj == -1) {
            t += dy
        } else if (this._top_bottom_adj == 1) {
            b += dy
        } else if (this._left_right_adj == 0) {
            t += dy
            b += dy
        }
        sx = (l + r)/2
        sy = (t + b)/2
        sw = max([5, Math.abs(r-l)])
        sh = max([5, Math.abs(t-b)])
        cds.data[xkey][index] = sx
        cds.data[ykey][index] = sy
        cds.data[wkey][index] = sw
        cds.data[hkey][index] = sh
        cds.data['l'][index] = sx - sw/2
        cds.data['t'][index] = sy - sh/2
      }
      cds.change.emit()
    }
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
      this._drag_points(ev, this.model.renderers)
    }
  }

  _pan_end(ev: PanEvent): void {
    this._pan(ev)
    if (ev.shiftKey) {
      const curpoint: [number, number] = [ev.sx, ev.sy]
      const [sx, sy] = this._compute_limits(curpoint)
      this._update(sx, sy, true)

      this.model.overlay.update({left: null, right: null, top: null, bottom: null})
      this._base_point = null
    } else {
      this._basepoint = null
      for (const renderer of this.model.renderers) {
        const cds = renderer.data_source
        this._emit_cds_changes(renderer.data_source, false, true, true)
        cds.properties.selected.change.emit()
      }
      this.h_axis_only = false
      this.v_axis_only = false

      if (this.pan_info != null)
        this.plot_view.push_state('pan', {range: this.pan_info})
    }
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

export namespace TsBboxEditTool {
  export type Attrs = p.AttrsOf<Props>

  export type Props = EditTool.Props & {
    dimensions: p.Property<Dimensions>
    default_cls: p.Property<string>
    num_objects: p.Property<number>
    renderers: p.Property<(GlyphRenderer & HasRectCDS)[]>
    overlay: p.Property<BoxAnnotation>
    match_aspect: p.Property<boolean>
  }
}

export interface TsBboxEditTool extends TsBboxEditTool.Attrs {}

export class TsBboxEditTool extends EditTool {
  properties: TsBboxEditTool.Props

  renderers: (GlyphRenderer & HasRectCDS)[]
  /*override*/ overlay: BoxAnnotation

  constructor(attrs?: Partial<TsBboxEditTool.Attrs>) {
    super(attrs)
  }

  static init_TsBboxEditTool(): void {
    this.prototype.default_view = TsBboxEditToolView

    this.define<TsBboxEditTool.Props>({
      dimensions: [ p.Dimensions, "both" ],
      default_cls: [ p.String, 'Other' ],
      num_objects: [ p.Int, 0 ],
      overlay: [ p.Instance,   DEFAULT_BOX_OVERLAY ],
      match_aspect: [ p.Boolean,    true],
    })
  }

  tool_name = "BoxEditTool\\ndouble tap: start/stop add bbox\\n"
              + "tap: select bbox or show x,y,pixel value\\nshift+tap: multi select\\n"
              + "left-dragging: move image or resize bbox\\n"
              + "shfit+left-dragging: zoom rectangular region\\n"
              + "BackSpace: delete the selected bbox\\nEsc: reset zoom\\n"
  icon = bk_tool_icon_box_edit
  event_type = ["tap" as "tap", "pan" as "pan", "move" as "move"]
  default_order = 1
}
"""


def get_bokeh_bbox_edit_app(img_path_list, source_list, labelset, desc_list, ncols=1,
                            max_size=1200, active_zoom=False, share_xy=False, label_level=False):
    from bokeh.core.validation.errors import INCOMPATIBLE_BOX_EDIT_RENDERER
    class TsBboxEditTool(EditTool, Drag, Tap):
        __implementation__ = TypeScript(JS_CODE)

        dimensions = Enum(Dimensions, default="both", help="""
        Which dimensions the box drawing is to be free in. By default, users may
        freely draw boxes with any dimensions. If only "width" is set, the box will
        be constrained to span the entire vertical space of the plot, only the
        horizontal dimension can be controlled. If only "height" is set, the box
        will be constrained to span the entire horizontal space of the plot, and the
        vertical dimension can be controlled.
        """)

        default_cls = String(default='Other')

        num_objects = Int(default=0, help="""
        Defines a limit on the number of boxes that can be drawn. By default there
        is no limit on the number of objects, but if enabled the oldest drawn box
        will be dropped to make space for the new box being added.
        """)

        @error(INCOMPATIBLE_BOX_EDIT_RENDERER)
        def _check_compatible_renderers(self):
            incompatible_renderers = []
            for renderer in self.renderers:
                if not isinstance(renderer.glyph, Rect):
                    incompatible_renderers.append(renderer)
            if incompatible_renderers:
                glyph_types = ', '.join(type(renderer.glyph).__name__ for renderer in incompatible_renderers)
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

            r = plot.rect('x', 'y', 'w', 'h', source=source, color='color',
                          line_alpha=0.7, fill_alpha=0.02,
                          selection_fill_alpha=0.0,
                          selection_line_alpha=1.0,
                          selection_color='color',
                          nonselection_fill_alpha=0.1,
                          nonselection_line_alpha=0.5,
                          nonselection_color='color')

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

            edit_tool = TsBboxEditTool(renderers=[r], default_cls=menu[0][1])

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
                level_dropdown = Dropdown(label="Level:",
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
