import torch
import plotly.graph_objects as go

def create_misclassification_plot(cm, classes):

    # Select initial class to show
    idx = 23
    initial_fp_count = sum(cm[idx][j] for j, val in enumerate(cm[idx, :]) if j != idx and val > 0)
    initial_fn_count = sum(cm[j][idx] for j, val in enumerate(cm[:, idx]) if j != idx and val > 0)

    fig = go.Figure()
    buttons = []
    for i, cls in enumerate(classes):
        # Calculate FP and FN total
        fp_sum = sum(cm[i][j] for j, val in enumerate(cm[i, :]) if j != i and val > 0)
        fn_sum = sum(cm[j][i] for j, val in enumerate(cm[:, i]) if j != i and val > 0)

        # Calculate false positives
        fp_classes = []
        fp_values = []
        for j, val in enumerate(cm[i, :]):  # Row represents predictions
            if j != i and val > 0: # Exclude diagonal (true positives) and zero values
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
        trace_FP = go.Scatter(
                x=[classes[j] for j in fp_classes],
                y=fp_values,
                mode='markers',
                name=f'FP ({cls})',
                marker=dict(symbol='diamond', size=20, color='blue'),
                visible=(i == idx)
            )

        # scatter plot trace for false negatives
        trace_NP = go.Scatter(
                x=[classes[j] for j in fn_classes],
                y=fn_values,
                mode='markers',
                name=f'FN ({cls})',
                marker=dict(symbol='circle', size=20, color='green'),
                visible=(i == idx)
            )

        # adding traces to fig
        fig.add_trace(trace_NP)
        fig.add_trace(trace_FP)

        # Create visibility list for traces
        visible = [False] * len(classes) * 2
        visible[i*2] = True    # Show false positives for selected class
        visible[i*2+1] = True  # Show false negatives for selected class

        button = dict(
            label=f'{cls} (TP: {cm[i][i]}, FP: {fp_sum}, FN: {fn_sum})',
            method='update',
            args=[
                {'visible': visible},
                {'title': f'Misclassifications for {cls}<br>True Positives = {cm[i][i]}, False Positives = {fp_sum}, False Negatives = {fn_sum}'}
            ]
        )
        buttons.append(button)

    # Update layout with scrollable axes
    fig.update_layout(
        title=dict(
            text=f'Misclassifications for {classes[idx]}<br>True Positives = {cm[idx][idx]}, False Positives = {initial_fp_count}, False Negatives = {initial_fn_count}',
            x=0.5,
            y=0.95,
        ),
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
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
            'xanchor': 'center',
            'yanchor': 'top',
            'x': 0.5,
            'y': 1.15,
        }],
        width=1200,
        height=800,
        margin=dict(l=50, r=50, b=100, t=150),  # Increased top margin t for legend
    )

    return fig


if __name__ == "__main__":

    classes = ['Akashiwo', 'Bacillaria', 'Bidulphia', 'Cochlodinium', 'Didinium_sp', 'Euplotes_sp', 'Eutintinnus',
               'Favella', 'Helicostomella_subulata', 'Hemiaulus', 'Karenia', 'Odontella', 'Parvicorbicula_socialis',
               'Pleuronema_sp', 'Protoperidinium', 'Stenosemella_sp1', 'Stephanopyxis', 'Strombidium_capitatum',
               'Tiarina_fusus', 'Tintinnid', 'Tintinnidium', 'bubble', 'pollen', 'zooplankton']
    
    cm = torch.tensor([
        [ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
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

    cm = cm.numpy()
    
    fig = create_misclassification_plot(cm, classes)
    fig.show()

