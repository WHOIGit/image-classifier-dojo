import os
import json
from typing import Optional, Literal

import aim
import plotly.graph_objects as go
import h5py as h5
import numpy as np
from scipy.io import savemat
import lightning.pytorch as pl
from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_reduce


class SaveValidationResults(pl.callbacks.Callback):

    def __init__(self, outdir, outfile, series, best_only=True):
        self.outdir = outdir
        self.outfile = outfile
        self.series = series
        self.best_only = best_only

    def on_validation_end(self, trainer, pl_module):
        log = trainer.callback_metrics
        metrics = pl_module.metrics
        # log: val_loss train_loss is_best f1micro
        # metric: {f1,recall,precision}_{weighted,micro,macro,perclass}, confusion_matrix{_norm}
        # pl_module.validation_{preds,targets,sources}

        is_best = bool(pl_module.best_epoch==pl_module.current_epoch)
        if not(is_best or not self.best_only):
            return

        current_epoch = int(pl_module.current_epoch)
        class_labels = pl_module.hparams.classes
        class_idxs = list(range(len(class_labels)))

        val_dataset = trainer.val_dataloaders.dataset
        train_dataset = trainer.train_dataloader.dataset
        val_counts_perclass = list(val_dataset.count_perclass.values())
        train_counts_perclass = list(train_dataset.count_perclass.values())
        counts_perclass = list(trainer.datamodule.count_perclass.values())
        training_image_fullpaths = train_dataset.sources
        training_image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in training_image_fullpaths]
        training_classes = train_dataset.targets

        output_scores = pl_module.validation_preds
        output_winscores = np.max(output_scores, axis=1)
        output_classes = np.argmax(output_scores, axis=1)
        input_classes = pl_module.validation_targets
        image_fullpaths = pl_module.validation_sources
        image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in image_fullpaths]

        assert output_scores.shape[0] == len(input_classes), 'wrong number inputs-to-outputs'
        assert output_scores.shape[1] == len(class_labels), 'wrong number of class labels'

        # STATS!
        stats = {}
        for mode in ['weighted', 'micro', 'macro', None]:
            for stat in ['f1', 'recall', 'precision']:
                key = f'{stat}_{mode or "perclass"}'  # f1|recall|precision _ micro|macro|weighted|perclass
                stats[key] = metrics[key].compute().cpu().numpy()

        # Classes order by some Stat
        classes_by = dict()
        #print(class_idxs, min(class_idxs), max(class_idxs))
        #print(counts_perclass)
        #print('min:',counts_perclass[min(class_idxs)])
        #print('len and last5', len(counts_perclass), counts_perclass[-5:])
        #print('max:', counts_perclass[max(class_idxs)])
        classes_by['count'] = sorted(class_idxs, key=lambda idx: counts_perclass[idx], reverse=True) #higher is better
        for stat in ['f1','recall','precision']:
            stat_perclass =  stats[stat+'_perclass']
            classes_by[stat] = sorted(class_idxs, key=lambda idx: (stat_perclass[idx]), reverse=True)

        # Confusion matrix
        confusion_matrix = metrics['confusion_matrix'].compute().cpu().numpy()
        confusion_matrix_norm = metrics['confusion_matrix_norm'].compute().cpu().numpy()

        ## PASSING IT DOWN TO OUTPUTS ##

        # default values
        results = dict(model_id=pl_module.hparams.model_id,
                       timestamp=pl_module.hparams.cmd_timestamp,
                       epoch = current_epoch,
                       class_labels=class_labels,
                       input_classes=input_classes,
                       output_classes=output_classes)

        # optional values
        if not self.best_only: results['is_best'] = is_best
        if 'image_fullpaths' in self.series: results['image_fullpaths'] = image_fullpaths
        if 'image_basenames' in self.series: results['image_basenames'] = image_basenames
        if 'training_image_fullpaths' in self.series: results['training_image_fullpaths'] = training_image_fullpaths
        if 'training_image_basenames' in self.series: results['training_image_basenames'] = training_image_basenames
        if 'training_classes' in self.series: results['training_classes'] = training_classes
        if 'output_winscores' in self.series: results['output_winscores'] = output_winscores
        if 'output_scores' in self.series: results['output_scores'] = output_scores
        if 'confusion_matrix' in self.series: results['confusion_matrix'] = confusion_matrix
        if 'confusion_matrix_norm' in self.series: results['confusion_matrix_norm'] = confusion_matrix_norm
        if 'counts_perclass' in self.series: results['counts_perclass'] = counts_perclass
        if 'val_counts_perclass' in self.series: results['val_counts_perclass'] = val_counts_perclass
        if 'train_counts_perclass' in self.series: results['val_counts_perclass'] = val_counts_perclass

        # optional stats and class_by's
        for stat in stats.keys(): # eg f1_weighted, recall_perclass
            if stat in self.series: results[stat] = stats[stat]
        for stat in classes_by:
            classes_by_stat = 'classes_by_'+stat
            if classes_by_stat in self.series: results[classes_by_stat] = classes_by[stat]

        # sendit!
        outfile = os.path.join(self.outdir,self.outfile).format(epoch=current_epoch)
        if is_best or not self.best_only:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.save_validation_results(outfile, results)

    def save_validation_results(self, outfile, results):
        if outfile.endswith('.json'): self._save_validation_results_json(outfile,results)
        if outfile.endswith('.mat'): self._save_validation_results_mat(outfile,results)
        if outfile.endswith(('.h5','.hdf')): self._save_validation_results_hdf(outfile,results)

    def _save_validation_results_json(self,outfile,results):
        for series in results: # convert numpy arrays to list
            if isinstance(results[series], np.ndarray):
                results[series] = results[series].tolist()
        # write json file
        with open(outfile, 'w') as f:
            json.dump(results, f)

    def _save_validation_results_mat(self,outfile,results):
        # index ints
        idx_data = ['input_classes','output_classes','training_classes']
        idx_data += ['classes_by_'+stat for stat in 'f1 recall precision count'.split()]
        str_data = ['class_labels','image_fullpaths','image_basenames','training_image_fullpaths','training_image_basenames']

        for series in results:
            if isinstance(results[series], np.ndarray): results[series] = results[series].astype('f4')
            elif isinstance(results[series], np.float64): results[series] = results[series].astype('f4')
            elif series in str_data: results[series] = np.asarray(results[series], dtype='object')
            elif series in idx_data: results[series] = np.asarray(results[series]).astype('u4') + 1
            # matlab is not zero-indexed, so increment all the indicies by 1

        savemat(outfile, results, do_compression=True)

    def _save_validation_results_hdf(self,outfile,results):
        attrib_data = ['model_id', 'timestamp', 'epoch']
        attrib_data += sum([f'f1_{mode} recall_{mode} precision_{mode}'.split() for mode in ('weighted','micro','macro')], [])
        int_data = ['input_classes', 'output_classes', 'training_classes']
        int_data += 'counts_perclass val_counts_perclass train_counts_perclass'.split()
        int_data += ['classes_by_'+stat for stat in 'f1 recall precision count'.split()]
        string_data = ['class_labels', 'image_fullpaths', 'image_basenames', 'training_image_fullpaths', 'training_image_basenames']
        with h5.File(outfile, 'w') as f:
            meta = f.create_dataset('metadata', data=h5.Empty('f'))
            for series in results:
                if series in attrib_data: meta.attrs[series] = results[series]
                elif series in string_data: f.create_dataset(series, data=np.string_(results[series]), compression='gzip', dtype=h5.string_dtype())
                elif series in int_data: f.create_dataset(series, data=results[series], compression='gzip', dtype='int16')
                elif isinstance(results[series],np.ndarray):
                    f.create_dataset(series, data=results[series], compression='gzip', dtype='float16')
                else: raise UserWarning('hdf results: WE MISSED THIS ONE: {}'.format(series))


class PlotValidationResults(pl.callbacks.Callback):

    def __init__(self, outdir, outfile, metric, best_only=True):
        self.outdir = outdir
        self.outfile = outfile
        self.metric = metric[0]
        self.best_only = best_only

    def on_validation_end(self, trainer, pl_module):
        log = trainer.callback_metrics
        metrics = pl_module.metrics
        # log: val_loss train_loss is_best f1macro
        # metric: {f1,recall,precision}_{weighted,micro,macro,perclass}, confusion_matrix{_norm}
        # pl_module .validation_{preds,targets,sources} .{training,validation}_loss_by_epoch

        #if not(log['is_best'] or not self.best_only):
        #    return

        current_epoch = pl_module.current_epoch
        class_labels = pl_module.hparams.classes
        class_idxs = list(range(len(class_labels)))

        # see https://lightning.ai/docs/torchmetrics/stable/pages/plotting.html
        fig, ax = metrics['confusion_matrix'].plot()
        fig.suptitle('HARK!')
        ax.set_title(f"a plot of {self.metric}")

        # sendit!
        outfile = os.path.join(self.outdir,self.outfile).format(epoch=current_epoch)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        fig.savefig(self.outfile)
        #if log['is_best'] or not self.best_only:
        #    os.makedirs(os.path.dirname(outfile), exist_ok=True)
        #    fig.savefig(self.outfile)
        return fig


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

        classes = trainer.datamodule.classes
        categories = [f'{c} {i:>03}' for i,c in enumerate(classes)]
        scores = metric_obj.compute().cpu().numpy()
        counts = trainer.datamodule.validation_dataset.count_perclass.values()

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
        on_hover = [f' Score: {s}<br>Counts: {c}' for c,s in zip(sorted_counts,sorted_scores)]

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
        context = dict(figure_order = self.order_by)
        [logger.experiment.track(aim_figure, name=name, epoch=epoch, context=context) for logger in trainer.loggers]


class PlotConfusionMetricAim(pl.callbacks.Callback):
    def __init__(self, title='Confusion Matrix (ep{EPOCH})',
                 normalize: Optional[Literal[True, "true", "pred", "all", "none", None]] = None,
                 order_by:str='recall_perclass', best_only=True):
        self.metric_key = 'confusion_matrix'
        self.normalize = 'true' if normalize is True else normalize
        self.title = title
        self.best_only = best_only
        self.order_by = order_by or 'classes'

    def on_validation_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if not(pl_module.best_epoch==epoch or not self.best_only):
            return

        metrics = pl_module.metrics
        metric_obj = metrics[self.metric_key]

        classes = trainer.datamodule.classes
        counts = trainer.datamodule.validation_dataset.count_perclass.values()
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
                rows.append(hovertext)
            html_cells.append(rows)

        # Create the heatmap plot using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=ordered_matrix,
            x=ordered_categories_x,
            y=ordered_categories,
            colorscale='Viridis',
            colorbar=dict(title='Count'),
            showscale=True,
            hovertext=html_cells,
            hoverinfo='text'
        ))

        # Calculate height based on the number of categories
        title = self.title.format(METRIC=self.metric_key, EPOCH=epoch, ORDER=order_by)
        height = 400 + len(categories) * 10  # base height + additional height per category

        # Customize the layout
        fig.update_layout(
            title=title,
            yaxis_title='Validation Actuals',
            xaxis_title='Validation Predicteds',
            height=height,
            width=height,
            font=dict(family='Courier New, monospace', size=10),
            hoverlabel=dict(font_family="Courier New, monospace", font_size=12),
            yaxis=dict(visible=True, autorange='reversed'),
        )

        aim_figure = aim.Figure(fig)
        #name, context = trainer.logger.parse_context(self.metric_key)
        name = self.metric_key
        context = dict(figure_order = self.order_by, normalize=self.normalize)
        [logger.experiment.track(aim_figure, name=name, epoch=epoch, context=context) for logger in trainer.loggers]