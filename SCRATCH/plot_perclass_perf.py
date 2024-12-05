import torch
from torch import Tensor
import plotly.graph_objects as go

epoch = 999
classes = ['Akashiwo', 'Bacillaria', 'Bidulphia', 'Cochlodinium', 'Didinium_sp', 'Euplotes_sp', 'Eutintinnus',
           'Favella', 'Helicostomella_subulata', 'Hemiaulus', 'Karenia', 'Odontella', 'Parvicorbicula_socialis',
           'Pleuronema_sp', 'Protoperidinium', 'Stenosemella_sp1', 'Stephanopyxis', 'Strombidium_capitatum',
           'Tiarina_fusus', 'Tintinnid', 'Tintinnidium', 'bubble', 'pollen', 'zooplankton']
categories = [f'{i:>03} {c}' for i,c in enumerate(classes)]

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
# For starters lets pick a single class, "Didinium_sp", and create a non-dropdown plot for that

focus_class = 'Didinium_sp'
idx = classes.index(focus_class)
TP_sum = cm[idx][idx]  # tensor notation
print(TP_sum)  # 7

fig = go.Figure()

#trace_FP = go.Scatter(...)
#trace_FN = go.Scatter(...)

#fig.add_trace(trace_FP)
#fig.add_trace(trace_FN)

# Customize the layout
fig.update_layout(
    title=f'{focus_class}; TP={TP_sum}; Epcoh',
    xaxis_title='Misclassifications Num',
    yaxis_title='Misclassification Classes',
    font = dict(family='Courier New, monospace', size=10),
)

fig.show()