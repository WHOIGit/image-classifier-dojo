import os
import json
from typing import Optional, Literal

import aim
import plotly.graph_objects as go
import h5py as h5
import numpy as np
from aim.sdk.adapters.pytorch_lightning import AimLogger
from scipy.io import savemat
import lightning.pytorch as pl
from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_reduce


class LogNormalizedLoss(pl.callbacks.Callback):
    def __init__(self, train_normloss='train_normloss', val_normloss='val_normloss'):
        self.train_normloss_name = train_normloss
        self.val_normloss_name = val_normloss

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.training_loss_by_epoch and pl_module.training_loss_by_epoch[0]: # else errors on autobatch
            train_normloss = pl_module.training_loss_by_epoch[pl_module.current_epoch]/pl_module.training_loss_by_epoch[0]
            pl_module.log('train_normloss', train_normloss, on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module,'validation_loss_by_epoch') and pl_module.validation_loss_by_epoch and pl_module.validation_loss_by_epoch[0]:
            val_normloss = pl_module.validation_loss_by_epoch[pl_module.current_epoch]/pl_module.validation_loss_by_epoch[0]
            pl_module.log('val_normloss', val_normloss, on_epoch=True)


class BarPlotMetricAim(pl.callbacks.Callback):
    def __init__(self, metric, title='{METRIC} (ep{EPOCH})', best_only=True, order_by:str='classes', order_reverse=False):
        self.metric_key = metric
        self.title = title
        self.best_only = best_only
        self.order_by = order_by
        self.order_reverse = order_reverse

    def on_validation_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if not(pl_module.best_epoch==epoch or not self.best_only):
            return

        metrics = pl_module.metrics
        metric_obj = metrics[self.metric_key]
        validation_dataset = trainer.datamodule.validation_dataset if trainer.datamodule else trainer.val_dataloaders.dataset
        classes = validation_dataset.classes
        categories = [f'{c} {i:>03}' for i,c in enumerate(classes)]
        scores = metric_obj.compute().cpu().numpy()
        counts = validation_dataset.count_perclass.values()

        if self.order_by.lower() in ['class', 'classes', 'classlist']:
            order_by = [i for i,c in enumerate(classes)]
        elif self.order_by.lower() == 'alphabetical':
            order_by = [c for i,c in enumerate(classes)]
        elif 'count' in self.order_by:
            order_by = counts
        elif self.order_by == self.metric_key:
            order_by = scores
        else:
            order_by = metrics[self.order_by].compute().cpu()

        sorted_categories,sorted_scores,sorted_counts,sorted_orderby = zip(*sorted(zip(categories, scores, counts, order_by), key=lambda x: x[-1], reverse=self.order_reverse))
        on_hover = [f' Score: {s:.4f}<br>Counts: {c}' for c,s in zip(sorted_counts,sorted_scores)]

        # Create horizontal bar graph
        fig = go.Figure(go.Bar(
            y=sorted_categories,  # categories on y-axis
            x=sorted_scores,  # values on x-axis
            orientation='h',  # horizontal bars
            text=[f'{score:.3f}' for score in sorted_scores],
            hovertext=on_hover,
            hoverinfo='text',
        ))

        # Calculate height based on the number of categories
        title = self.title.format(METRIC=self.metric_key, EPOCH=epoch, ORDER=self.order_by)
        height = 400 + len(categories) * 12  # base height + additional height per category

        # Customize the layout
        fig.update_layout(
            title=title,
            xaxis_title=self.metric_key,
            yaxis_title='Classes',
            height=height,     # Adjust height based on number of categories
            font = dict(family='Courier New, monospace',size=10),
            hoverlabel = dict(font_family="Courier New, monospace", font_size=12),
        )


        aim_figure = aim.Figure(fig)
        #name, context = trainer.logger.parse_context(self.metric_key)
        name = self.metric_key
        context = dict(figure_order = self.order_by, subset='val')
        for logger in trainer.loggers:
            if isinstance(logger,AimLogger):
                 logger.experiment.track(aim_figure, name=name, epoch=epoch, context=context)


class PlotConfusionMetricAim(pl.callbacks.Callback):
    def __init__(self, title='Confusion Matrix (ep{EPOCH})',
                 normalize: Optional[Literal[True, "true", "pred", "all", "none", None]] = None,
                 order_by:str='recall_perclass', best_only=True, hide_zeros=True):
        self.metric_key = 'confusion_matrix'
        self.normalize = 'true' if normalize is True else normalize
        self.title = title
        self.best_only = best_only
        self.order_by = order_by or 'classes'
        self.hide_zeros = hide_zeros

    def on_validation_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if not(pl_module.best_epoch==epoch or not self.best_only):
            return

        metrics = pl_module.metrics
        metric_obj = metrics[self.metric_key]

        validation_dataset = trainer.datamodule.validation_dataset if trainer.datamodule else trainer.val_dataloaders.dataset
        classes = validation_dataset.classes
        counts = validation_dataset.count_perclass.values()
        matrix_counts = metric_obj.compute().cpu()
        categories = [f'({int(sum)}) {c} {i:>03}' for i, c, sum in zip(range(len(classes)), classes, matrix_counts.sum(dim=1))]
        categories_x = [f'{i:>03} {c} ({int(sum)})' for i, c, sum in zip(range(len(classes)), classes, matrix_counts.sum(dim=0))]

        order_reverse = False
        if self.order_by.lower() in ['class', 'classes', 'classlist']:
            order_by = [i for i,c in enumerate(classes)]
        elif self.order_by.lower() == 'alphabetical':
            order_by = [c for i,c in enumerate(classes)]
        elif 'count' in self.order_by:
            order_by = counts
            order_reverse = True
        else:
            order_by = metrics[self.order_by].compute().cpu()
            order_reverse = True

        matrix = _confusion_matrix_reduce(matrix_counts, self.normalize)

        ordered_classes, ordered_categories, ordered_categories_x, ordered_counts, ordered_orderby = \
            zip(*sorted(zip(classes, categories, categories_x, counts, order_by),
                        key=lambda x: x[-1], reverse=order_reverse))
        reordered_indices = [categories.index(c) for c in ordered_categories]

        ordered_matrix = matrix[reordered_indices, :][:, reordered_indices]
        ordered_matrix_count = matrix_counts[reordered_indices, :][:, reordered_indices]

        if self.hide_zeros:
            ordered_matrix[ordered_matrix==0] = np.nan

        html_cells = []
        recall = metrics['recall_perclass'].compute().cpu()
        precision = metrics['precision_perclass'].compute().cpu()
        f1score = metrics['f1_perclass'].compute().cpu()
        for i, class_actual in enumerate(ordered_classes):
            rows = []
            for j, class_predicted in enumerate(ordered_classes):
                lines = []
                cell_count = int(ordered_matrix_count[i, j])
                actuals_count = int(ordered_matrix_count.sum(dim=1)[i])
                predicteds_count = int(ordered_matrix_count.sum(dim=0)[j])
                lines.append(f'<b>    Count:</b> {cell_count}')
                lines.append(f'<b>   Actual:</b> {classes.index(class_actual):>03} {class_actual}')
                lines.append(f'<b>Predicted:</b> {classes.index(class_predicted):>03} {class_predicted}')
                if i==j:
                    class_idx = classes.index(class_actual)
                    lines.append(' ---------------- ')
                    lines.append(f'<b>   Recall:</b> {recall[class_idx]:.3f} ({cell_count} of {actuals_count})')
                    lines.append(f'<b>Precision:</b> {precision[class_idx]:.3f} ({cell_count} of {predicteds_count})')
                    lines.append(f'<b> F1 Score:</b> {f1score[class_idx]:.3f}')
                # TODO show images in cells
                #      or embed links
                hovertext = '<BR>'.join(lines)  # + f'<extra></extra>'
                if cell_count==0: hovertext=''
                rows.append(hovertext)
            html_cells.append(rows)

        # Create the heatmap plot using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=ordered_matrix,
            x=ordered_categories_x,
            y=ordered_categories,
            colorscale='Viridis',
            colorbar=dict(title='Recall' if self.normalize else 'Counts'),
            showscale=True,
            hovertext=html_cells,
            hoverinfo='text'
        ))

        # Calculate height based on the number of categories
        title = self.title.format(METRIC=self.metric_key, EPOCH=epoch, ORDER=order_by)
        height = 400 + len(categories) * 14  # base height + additional height per category

        # Customize the layout
        fig.update_layout(
            title=title,
            height=height,
            width=height,
            font=dict(family='Courier New, monospace', size=10),
            hoverlabel=dict(font_family="Courier New, monospace", font_size=12),
            yaxis=dict(title='Validation Actuals', visible=True, autorange='reversed', tickson='boundaries'),
            xaxis=dict(title='Validation Predicteds', visible=True, tickson='boundaries'),
        )

        aim_figure = aim.Figure(fig)
        #name, context = trainer.logger.parse_context(self.metric_key)
        name = self.metric_key
        context = dict(figure_order = self.order_by, normalize=self.normalize, subset='val')
        for logger in trainer.loggers:
            if isinstance(logger, AimLogger):
                logger.experiment.track(aim_figure, name=name, epoch=epoch, context=context)


class PlotPerclassDropdownAim(pl.callbacks.Callback):
    def __init__(self, title='Class Compare (ep{EPOCH})', best_only=True):
        self.metric_key = 'confusion_matrix'
        self.title = title
        self.best_only = best_only

    def on_validation_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if not(pl_module.best_epoch==epoch or not self.best_only):
            return

        metrics = pl_module.metrics
        cm = metrics['confusion_matrix'].compute().cpu()

        validation_dataset = trainer.datamodule.validation_dataset if trainer.datamodule else trainer.val_dataloaders.dataset
        classes = validation_dataset.classes
        categories_y = [f'{c} {i:>03}' for i,c,sum in zip(range(len(classes)), classes, cm.sum(dim=1))]
        categories = [f'{i:>03} {c}' for i,c,sum in zip(range(len(classes)), classes, cm.sum(dim=1))]


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
                    fp_values.append(val.item())

            # Calculate false negatives
            fn_classes = []
            fn_values = []
            for j, val in enumerate(cm[:, i]):  # Column represents actual classes
                if j != i and val > 0:  # Exclude diagonal (true positives) and zero values
                    fn_classes.append(j)
                    fn_values.append(val.item())

            # scatter plot trace for false positives
            trace_FP = go.Bar(
                y=[categories_y[j] for j in fp_classes],
                x=fp_values,
                name=f'FP ({cls})',
                marker=dict(color='blue'),
                orientation='h',
                visible=(i == idx),
                texttemplate='FP:%{x}',
                hovertext=[f'{v} "{classes[c]}"<br>{"where" if v>1 else "was"} misclassified-as<br>"{classes[i]}"' for c,v in zip(fp_classes,fp_values)],
                hoverinfo='text',
            )

            # scatter plot trace for false negatives
            trace_FN = go.Bar(
                y=[categories_y[j] for j in fn_classes],
                x=fn_values,
                name=f'FN ({cls})',
                marker=dict(color='green'),
                orientation='h',
                visible=(i == idx),
                texttemplate='FN:%{x}',
                hovertext=[f'{v} "{classes[i]}"<br>{"where" if v>1 else "was"} misclassified-as<br>"{classes[c]}"' for c, v in zip(fn_classes, fn_values)],
                hoverinfo='text',
            )

            # adding traces to fig
            fig.add_trace(trace_FN)
            fig.add_trace(trace_FP)

            # Create visibility list for traces
            visible = [False] * len(classes) * 2
            visible[i * 2] = True  # Show false positives for selected class
            visible[i * 2 + 1] = True  # Show false negatives for selected class

            button = dict(
                label=f'{categories[i]} (TP:{cm[i][i]:.0f} FP:{fp_sum:.0f} FN:{fn_sum:.0f})',
                method='update',
                args=[
                    {'visible': visible},
                    {
                        'title': f'Misclassifications for "{classes[i]}"<br>{cm[i][i]:.0f} True Positives, {fp_sum:.0f} False Positives, {fn_sum:.0f} False Negatives'}
                ]
            )
            buttons.append(button)

        # Update layout with scrollable axes
        fig.update_layout(
            barmode='stack',
            height=400+len(classes)*2,
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

        aim_figure = aim.Figure(fig)
        name = 'Misclassifieds Explorer'
        context = dict(subset='val')
        for logger in trainer.loggers:
            if isinstance(logger, AimLogger):
                logger.experiment.track(aim_figure, name=name, epoch=epoch, context=context)
