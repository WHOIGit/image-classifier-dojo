import torch
from torch import Tensor
import plotly.graph_objects as go

epoch = 999
classes = ['Akashiwo', 'Bacillaria', 'Bidulphia', 'Cochlodinium', 'Didinium_sp', 'Euplotes_sp', 'Eutintinnus',
           'Favella', 'Helicostomella_subulata', 'Hemiaulus', 'Karenia', 'Odontella', 'Parvicorbicula_socialis',
           'Pleuronema_sp', 'Protoperidinium', 'Stenosemella_sp1', 'Stephanopyxis', 'Strombidium_capitatum',
           'Tiarina_fusus', 'Tintinnid', 'Tintinnidium', 'bubble', 'pollen', 'zooplankton']

cm = Tensor([[ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  2],
             [ 0,  0,  0,  0,  7,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1],
             [ 0,  0,  0,  0,  0, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  1],
             [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3],
             [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 14,  0,  0,  2,  0,  0,  0,  0,  1,  0,  0,  0,  3],
             [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0, 18,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
             [ 0,  0,  1,  0,  5,  2,  0,  0,  0,  0,  0,  0,  0, 21,  1,  0,  1,  0,  1,  0,  0,  0,  0,  2],
             [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 27,  0,  0,  0,  0,  1,  0,  0,  0,  0],
             [ 0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0,  0,  2,  0,  0,  0,  4],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,  0, 12,  1,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0,  1],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  1],
             [ 0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  6,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  1],
             [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0],
             [ 0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1],
             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0, 22]])

# to convert cm to numpy array do:
cm = cm.cpu().numpy()

# Y axis are the ACTUALS, the true labels
# X axis are the PREDICTIONS, the output of the model
# top to bottom Y axis is in CLASSES order
# left to right X axis is in CLASSES order
# each item on the top-left to bottom-right DIAGONAL is a TP True Positive (the ACTUAL and PREDICTION match)
# for a given CLASS, each item in a ROW (that is not on the DIAGONAL) is a False Positive (FP)
#    FP: predicts a class when it’s actually another class
# for a given CLASS, each item in a COLUMN (that is not on the DIAGONAL) is a False Negative (FN)
#    FN: fails to predict a class when it’s actually present

# We want a plot with a dropdown-select to select a class to focus on.
# [optional] The dropdown labels show the sum of FPs and FNs to help guide the reviewer
# When a class is selected, the plot is updated to show the distribution of FP and FN for that class wrt other classes
# See https://plotly.com/python/dropdowns/
# see https://stackoverflow.com/a/56236906 for dropdown example
# For starters lets pick a single class, "Didinium_sp", and create a non-dropdown plot for that

categories_y = [f'{c} {i:>03}' for i, c in zip(range(len(classes)), classes)]
categories = [f'{i:>03} {c}' for i, c in zip(range(len(classes)), classes)]

# Select initial class to show
idx = 0
initial_fp_count = sum(cm[idx][j] for j, val in enumerate(cm[idx, :]) if j != idx and val > 0)
initial_fn_count = sum(cm[j][idx] for j, val in enumerate(cm[:, idx]) if j != idx and val > 0)

fig = go.Figure()
buttons = []
for i, cls in enumerate(categories_y):
    # Calculate FP and FN total
    fp_sum = sum(cm[i][j] for j, val in enumerate(cm[i, :]) if j != i and val > 0)
    fn_sum = sum(cm[j][i] for j, val in enumerate(cm[:, i]) if j != i and val > 0)

    # Calculate false positives
    fp_classes = []
    fp_values = []
    for j, val in enumerate(cm[i, :]):  # Row represents predictions
        if j != i and val > 0:  # Exclude diagonal (true positives) and zero values
            fp_classes.append(j)
            fp_values.append(val)

    # Calculate false negatives
    fn_classes = []
    fn_values = []
    for j, val in enumerate(cm[:, i]):  # Column represents actual classes
        if j != i and val > 0:  # Exclude diagonal (true positives) and zero values
            fn_classes.append(j)
            fn_values.append(val)


    # scatter plot trace for false positives
    trace_FP = go.Bar(
        y=[categories_y[j] for j in fp_classes],
        x=fp_values,
        name=f'FP ({cls})',
        marker=dict(color='blue'),
        orientation='h',
        visible=(i == idx),
    )

    # scatter plot trace for false negatives
    trace_NP = go.Bar(
        y=[categories_y[j] for j in fn_classes],
        x=fn_values,
        name=f'FN ({cls})',
        marker=dict(color='green'),
        orientation='h',
        visible=(i == idx),
    )

    # adding traces to fig
    fig.add_trace(trace_NP)
    fig.add_trace(trace_FP)

    # Create visibility list for traces
    visible = [False] * len(classes) * 2
    visible[i * 2] = True  # Show false positives for selected class
    visible[i * 2 + 1] = True  # Show false negatives for selected class

    button = dict(
        label=f'{categories[i]} (TP: {cm[i][i]:.0f}, FP: {fp_sum:.0f}, FN: {fn_sum:.0f})',
        method='update',
        args=[
            {'visible': visible},
            {'title': f'Misclassifications for {categories[i]}<br>True Positives = {cm[i][i]:.0f}, False Positives = {fp_sum:.0f}, False Negatives = {fn_sum:.0f}'}
        ]
    )
    buttons.append(button)

# Update layout with scrollable axes
fig.update_layout(
    barmode='stack',
    title=dict(
        text=f'Misclassifications for {categories[idx]}<br>True Positives = {cm[idx][idx]:.0f}, False Positives = {initial_fp_count:.0f}, False Negatives = {initial_fn_count:.0f}',
        x=0.5,
        y=0.95,
    ),
    hoverlabel=dict(namelength=-1, font_family="Courier New, monospace", font_size=12),
    font=dict(family='Courier New, monospace', size=10),
    yaxis=dict(
        visible=True,
        showgrid=True,
        autorange='reversed',
    ),
    xaxis=dict(
        title='Number of Misclassifications',
        showgrid=True,
        rangemode='tozero'
    ),
    hovermode='closest',
    showlegend=True,
    legend=dict(
        orientation="h",
        xanchor="right",
        yanchor="bottom",
        x=1,
        y=1.02,
    ),
    updatemenus=[{
        'active': idx,
        'buttons': buttons,
        'direction': 'down',
        'showactive': True,
        'xanchor': 'left',
        'yanchor': 'top',
        'y': 1.15,
    }],
)

fig.show()