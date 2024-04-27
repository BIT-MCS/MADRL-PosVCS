import plotly.graph_objs as go
from PIL import Image
import cv2 as cv


def save_figure_to_html(fig, filename):
    fig.write_html(filename)

def save_figure_to_png(fig, filename):
    fig.write_image(filename)

def visualize_floor_plan(floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar", title=None, show=False):
    # show plain map
    # add floor plan
    fig = go.Figure()
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])
    
    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        autosize=True,
        width=900,
        height=1035,
        template="plotly_white",
    )

    fig.update_xaxes(autorange=False, showgrid= False, zeroline = False, 
    visible = False ,range=[0, width_meter])
    fig.update_yaxes(autorange=False, showgrid= False, zeroline = False, 
    visible = False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    
    if show:
        fig.show()
    
    return fig

def visualize_trajectory(trajectory, floor_plan_filename, width_meter, height_meter, title=None, mode='lines + markers + text', show=False):
    fig = go.Figure()

    # add trajectory
    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    color_list = ['rgba(4, 174, 4, 0.5)'] * trajectory.shape[0]
    color_list[0] = 'rgba(12, 5, 235, 1)'
    color_list[-1] = 'rgba(235, 5, 5, 1)'

    position_count = {}
    text_list = []
    for i in range(trajectory.shape[0]):
        if str(trajectory[i]) in position_count:
            position_count[str(trajectory[i])] += 1
        else:
            position_count[str(trajectory[i])] = 0
        text_list.append('        ' * position_count[str(trajectory[i])] + f'{i}')
    text_list[0] = 'Start Point: 0'
    text_list[-1] = f'End Point: {trajectory.shape[0] - 1}'

    fig.add_trace(
        go.Scattergl(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode=mode,
            marker=dict(size=size_list, color=color_list),
            line=dict(shape='linear', color='rgb(100, 10, 100)', width=2, dash='dot'),
            text=text_list,
            textposition="top center",
            name='trajectory',
        ))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.layout.update(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig

def visualize_heatmap(position, value, floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar", title=None, show=False):
    fig = go.Figure()
    #print(value, 'value here')
    # add heat map
    fig.add_trace(
        go.Scatter(x=position[:, 0],
                   y=position[:, 1],
                   mode='markers',
                   marker=dict(size=7,
                               color=value,
                               colorbar=dict(title=colorbar_title),
                               colorscale="Rainbow"),
                   text=value,
                   name=title))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig

def visualize_position(position, value, floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar", title=None, show=False):
    fig = go.Figure()
    #print(value, 'value here!!!')
    # add heat map
    fig.add_trace(
        go.Scatter(x=position[:, 0],
                   y=position[:, 1],
                   mode='markers',
                   marker=dict(size = 2,
                               color='rgb(255,0,0)'
                               ),
                   text=value,
                   name=title))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    floor_plan = Image.open('./output/mod_floorplan/site2/F4_vis.png')
    print(cv.imread(floor_plan_filename).shape,'aaaaa')
    print(cv.imread('./output/mod_floorplan/site2/F4_vis.png').shape,'aaaaa')
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, showgrid= False, zeroline = False, 
    visible = False ,range=[0, width_meter])
    fig.update_yaxes(autorange=False, showgrid= False, zeroline = False, 
    visible = False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    
    fig.update_layout(
        showlegend = False,
        autosize=True,
        width=900,
        height=200 + 900 * height_meter / width_meter,
        # width = 742,
        # height = 800,
        template="plotly_white",
    )
    
    if show:
        fig.show()

    return fig
