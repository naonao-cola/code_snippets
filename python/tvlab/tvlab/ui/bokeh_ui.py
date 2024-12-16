'''
Copyright (C) 2023 TuringVision

Bokeh plot interface
'''

import numpy as np
from math import pi
import bokeh
from tvlab.utils import IN_NOTEBOOK

if IN_NOTEBOOK:
    from bokeh.io import output_notebook
    from bokeh.resources import INLINE
    output_notebook(INLINE)

BOKEH_VERSION = float('.'.join(bokeh.__version__.split('.')[:2]))

__all__ = ['COLOR_MAP', 'get_one_color', 'plot_table', 'plot_lines',
           'plot_stack_bar', 'plot_bar', 'plot_bokeh_matrix',
           'plot_bokeh_scatter', 'plot_bokeh_histogram', 'bokeh_figs_to_html']

COLOR_MAP = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#000000"]


def _format_legend(legend):
    legend.label_text_font_size = '6pt'
    legend.border_line_alpha = 0.2
    legend.glyph_height = 10
    legend.glyph_width = 10
    legend.label_height = 10
    legend.label_width = 25
    legend.padding = 5


def get_one_color(index):
    index = index % len(COLOR_MAP)
    return COLOR_MAP[index]


def plot_table(columns_data, columns_fmt=None, title=None, need_show=True):
    """ Plot table with bokeh

    # Arguments
        columns_data: (dict)
            display data
        columns_fmt: (dict)
            display format
        need_show: bool
    # return
        p: bokeh figure
    """
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter, Div
    from bokeh.layouts import Column
    from bokeh.plotting import show
    __ROW_HEIGHT__ = 25

    source = ColumnDataSource(columns_data)

    total_height = __ROW_HEIGHT__ * (1 + len(list(columns_data.values())[0])) + 2

    columns = []
    for field in columns_data.keys():
        fmt = None
        if columns_fmt and field in columns_fmt:
            fmt = NumberFormatter(format=columns_fmt[field])
        one_column = TableColumn(field=field, formatter=fmt, title=field)
        columns.append(one_column)

    if BOKEH_VERSION >= 1.4:
        p = DataTable(source=source,
                      columns=columns,
                      width_policy='max',
                      height=total_height)
        if title:
            title = Div(text=title, width_policy='max')
            p = Column(title, p, width_policy='max')
    else:
        p = DataTable(source=source,
                      columns=columns,
                      width=-1,
                      height=total_height)
        if title:
            title = Div(text=title, width=-1)
            p = Column(title, p)
    if need_show:
        show(p)
    return p


def plot_lines(title, xdata, ydata, x_range=None, y_range=None, xlabel='x', ylabel='y', need_show=True):
    ''' Plot lines with bokeh

    # Arguments
        xdata: list or dict
        ydata: list or dict
    '''
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure, show
    tools = 'hover, pan, box_zoom, wheel_zoom, reset, save'
    p = figure(title=title, background_fill_color="white",
               plot_width=850,
               x_range=x_range, y_range=y_range, tools=tools)

    line_keys = [None]
    if isinstance(xdata, dict):
        line_keys = list(xdata.keys())
    elif isinstance(ydata, dict):
        line_keys = list(ydata.keys())

    for i, key in enumerate(line_keys):
        x = xdata if not isinstance(xdata, dict) else xdata[key]
        y = ydata if not isinstance(ydata, dict) else ydata[key]

        legend = str(key)
        source_curve = ColumnDataSource(data=dict(x_values=x, y_values=y,
                                                  legend=[str(key)] * len(x)))
        if BOKEH_VERSION >= 1.4:
            p.line('x_values', 'y_values', source=source_curve,
                   line_color=get_one_color(i),
                   line_width=1, legend_label=legend)
        else:
            p.line('x_values', 'y_values', source=source_curve,
                   line_color=get_one_color(i),
                   line_width=1, legend=legend)

    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    p.legend.click_policy = "hide"
    p.select_one(HoverTool).tooltips = [('label', '@legend'),
                                        (ylabel, "@y_values"),
                                        (xlabel, "@x_values")]
    _format_legend(p.legend)
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = - pi / 4

    if need_show:
        show(p)
    return p


def plot_stack_bar(title, data, xlabel, ylabel, need_show=True):
    ''' Plot stacked bar with bokeh

    # Arguments
        data: dict
            eg: {'x': ['a', 'b', 'c'], 'y1':[1, 2, 3], 'y2':[2,3,4]}
        xlabel: key in dict
        ylabel: key in dict
    '''
    from bokeh.plotting import figure, show
    from bokeh.core.properties import value
    if isinstance(ylabel, str):
        ylabel = [ylabel]

    tools = 'hover, pan, box_zoom, wheel_zoom, reset, save'
    p = figure(x_range=data[xlabel], plot_height=300, title=title,
               tools=tools, tooltips="$name @{}: @$name".format(xlabel))

    colors = [get_one_color(i) for i, _ in enumerate(ylabel)]

    if BOKEH_VERSION >= 1.4:
        p.vbar_stack(ylabel, x=xlabel, width=0.6, source=data, color=colors,
                     legend_label=ylabel, line_color=None)
    else:
        p.vbar_stack(ylabel, x=xlabel, width=0.6, source=data, color=colors,
                     legend=[value(x) for x in ylabel], line_color=None)
    _format_legend(p.legend)
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = - pi / 4
    if need_show:
        show(p)
    return p


def plot_bar(title, data, xlabel, ylabel, group_name=None, need_show=True):
    ''' Plot bar with bokeh

    # Arguments
        data: dict
            eg: {'x': ['a', 'b', 'c'], 'y':[1, 2, 3]}
        xlabel: key in dict
        ylabel: key in dict
        group_name: dict
            eg: {'xxx': ['a', 'c'], 'kkk': ['b']}
    '''
    from bokeh.plotting import figure, show
    p = figure(x_range=data[xlabel], plot_height=300, title=title,
               toolbar_location=None, tools="hover, save",
               tooltips="$name @{}: @$name".format('xlabel'))

    if group_name is None:
        colors = [get_one_color(i) for i, _ in enumerate(data[xlabel])]
        p.vbar(data[xlabel], width=0.5, bottom=0, top=data[ylabel],
               color=colors)
    else:
        for i, key in enumerate(group_name.keys()):
            group = group_name[key]
            group_value = [data[ylabel][data[xlabel].index(x)] for x in group]
            if BOKEH_VERSION >= 1.4:
                p.vbar(group, width=0.5, bottom=0, top=group_value,
                       color=get_one_color(i), legend_label=key)
            else:
                p.vbar(group, width=0.5, bottom=0, top=group_value,
                       color=get_one_color(i), legend=key)
        _format_legend(p.legend)
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = - pi / 4
    if need_show:
        show(p)
    return p


def plot_bokeh_scatter(title,
                       x_range, y_range,
                       x, y,
                       x_label=None,
                       y_label=None,
                       data=None,
                       need_show=True):
    ''' Plot scatter with bokeh

    # Arguments
        title:
        x_range:
        y_range:
        x: key in data
        y: key in data
        x_label: str
        y_label: str
        data: eg[{'x': 1.0, 'y': 2.0, 'color':$32023, 'alpha': 0.8, 'size':3} ...]
    '''
    import pandas as pd
    from bokeh.plotting import figure, show
    from bokeh.transform import jitter
    pd_data = pd.DataFrame(data)
    p = figure(plot_width=800, plot_height=len(y_range) * 50 + 50,
               y_range=y_range, x_range=x_range, title=title)

    p.circle(x=x, y=jitter(y, width=0.6, range=p.y_range),
             color='color', size='size', source=pd_data, alpha='alpha')
    p.ygrid.grid_line_color = None
    p.yaxis.axis_label = y_label
    p.xaxis.axis_label = x_label
    if need_show:
        show(p)
    return p


def plot_bokeh_matrix(title,
                      x_labels=None,
                      y_labels=None,
                      colors=None,
                      alphas=None,
                      texts=None,
                      tips=None,
                      cell_size=(50, 50),
                      need_show=True):
    ''' Plot matrix with bokeh

    # Arguments
        x_labels: list
        y_labels: list
        texts: list (size = len(x_labels) * len(y_labels))
        tips: list (size = len(texts)
    '''
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure, show
    tools = 'hover, save'
    p = figure(x_range=x_labels,
               y_range=y_labels[::-1],
               plot_height=cell_size[1] * len(y_labels) + 100,
               plot_width=cell_size[0] * len(x_labels) + 100,
               x_axis_location="above",
               y_axis_location="left",
               title=title, tools=tools)
    x_name = x_labels * len(y_labels)
    y_name = [y for y in y_labels for _ in range(len(x_labels))]
    if colors is None:
        colors = ["#00ff7f"] * len(x_name)
    elif not isinstance(colors, (tuple, list)):
        colors = [colors] * len(x_name)
    if alphas is None:
        alphas = [1] * len(x_name)

    source = ColumnDataSource(dict(x_name=x_name, y_name=y_name,
                                   colors=colors, alphas=alphas,
                                   texts=texts, tips=tips))
    p.rect('x_name',
           'y_name',
           0.9,
           0.9,
           source=source,
           color='colors',
           line_color=None,
           alpha='alphas')

    p.text('x_name', 'y_name', 'texts', source=source,
           text_align="center",
           text_baseline="middle",
           text_font_size='8pt')
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = - pi / 4

    p.select_one(HoverTool).tooltips = [('Percent', '@tips')]
    if need_show:
        show(p)
    return p


def plot_bokeh_histogram(title, data, xlabel, ylabel, xbins=20, ybins=20, need_show=True):
    ''' Plot histogram

    # Arguments
        data: dict
            eg: {'x': [2.0, 1.5, 3.2, ...], 'y':[1.5, 2.1, 5.9, ...]}
        xlabel (str): key in dict
        ylabel (str): key in dict
        xbins (int):
        ybins (int):
    '''
    from bokeh.models import CustomJS, BoxSelectTool, LassoSelectTool
    from bokeh.plotting import figure, show
    from bokeh.layouts import gridplot

    TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"

    # create the scatter plot
    p = figure(tools=TOOLS, plot_width=500, plot_height=500, min_border=10, min_border_left=50,
               toolbar_location="above", x_axis_location=None, y_axis_location=None,
               title=title)
    p.background_fill_color = "#fafafa"
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False

    r = p.scatter(x=xlabel, y=ylabel, source=data, size=3, color="#3A5785", alpha=0.6)

    # create the horizontal histogram
    hhist, hedges = np.histogram(data[xlabel], bins=xbins)
    hzeros = np.zeros(len(hedges)-1)
    hmax = max(hhist)*1.1

    LINE_ARGS = dict(color="#3A5785", line_color=None)

    ph = figure(toolbar_location='above', plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
                y_range=(-hmax, hmax), min_border=10, min_border_left=50,
                x_axis_location='above', y_axis_location='right')
    ph.xgrid.grid_line_color = None
    ph.yaxis.major_label_orientation = np.pi/4
    ph.background_fill_color = "#fafafa"
    ph.xaxis.axis_label = xlabel

    ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
    hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
    hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

    # create the vertical histogram
    vhist, vedges = np.histogram(data[ylabel], bins=ybins)
    vzeros = np.zeros(len(vedges)-1)
    vmax = max(vhist)*1.1

    pv = figure(plot_width=200, plot_height=p.plot_height, x_range=(-vmax, vmax),
                y_range=p.y_range, min_border=10, y_axis_location='left')
    pv.ygrid.grid_line_color = None
    pv.xaxis.major_label_orientation = np.pi/4
    pv.background_fill_color = "#fafafa"
    pv.yaxis.axis_label = ylabel

    pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="#3A5785")
    vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
    vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

    layout = gridplot([[p, pv], [ph, None]], merge_tools=False)
    callback = CustomJS(args=dict(s=r.data_source,
                                  xlabel=xlabel, ylabel=ylabel,
                                  hedges=hedges, vedges=vedges,
                                  hh1=hh1, hh2=hh2,
                                  vh1=vh1, vh2=vh2), code="""
        const inds = s.selected.indices;
        var x = s.data[xlabel];
        var y = s.data[ylabel];

        var hhist1 = [];
        var hhist2 = [];
        var vhist1 = [];
        var vhist2 = [];

        for (var j = 1; j < hedges.length; j ++) {
            hhist1.push(0);
            hhist2.push(0);
        }

        for (var j = 1; j < vedges.length; j ++) {
            vhist1.push(0);
            vhist2.push(0);
        }

        if (((inds.length != 0) && (inds.length != x.length))) {
            for (var i = 0; i < inds.length; i++) {
                for (var j = 1; j < hedges.length; j ++) {
                    if (x[inds[i]] < hedges[j]) {
                        hhist1[j-1]++;
                        break;
                    }
                }

                for (var j = 1; j < vedges.length; j ++) {
                    if (y[inds[i]] < vedges[j]) {
                        vhist1[j-1]++;
                        break;
                    }
                }
            }
            for (var i = 0; i < x.length; i++) {
                if (!inds.includes(i)) {
                    for (var j = 1; j < hedges.length; j ++) {
                        if (x[i] < hedges[j]) {
                            hhist2[j-1]--;
                            break;
                        }
                    }

                    for (var j = 1; j < vedges.length; j ++) {
                        if (y[i] < vedges[j]) {
                            vhist2[j-1]--;
                            break;
                        }
                    }
                }
            }
        }


        console.log('hhist2:', hhist2.length, 'ori hist2:', hh2.data_source.data["top"].length);
        hh1.data_source.data["top"] =  hhist1;
        hh2.data_source.data["top"]   = hhist2;
        vh1.data_source.data["right"] =  vhist1;
        vh2.data_source.data["right"] = vhist2;
        hh1.data_source.change.emit();
        hh2.data_source.change.emit();
        vh1.data_source.change.emit();
        vh2.data_source.change.emit();
    """)

    r.data_source.selected.js_on_change('indices', callback)
    if need_show:
        show(layout)
    return layout


def bokeh_figs_to_html(fig_list, html_path='./index.html', title=''):
    from bokeh.plotting import save
    from bokeh.layouts import Column
    from bokeh.resources import INLINE
    save(Column(*fig_list), filename=html_path, resources=INLINE, title=title)
