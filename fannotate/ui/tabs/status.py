import gradio as gr
from gradio_rich_textbox import RichTextbox
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from bert_score import BERTScorer
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib
matplotlib.use('Agg')

def create_status_tab(annotator, demo=None):
    """Creates and returns the status tab interface"""
    with gr.Tab("📊 Analysis"):
        with gr.Row():
            gr.Markdown("## Agreement Analysis")
        
        with gr.Row():
            # Left column for controls
            with gr.Column(scale=1):
                category_select = gr.Dropdown(
                    label="Select Category to Analyze",
                    choices=[],
                    interactive=True
                )
            with gr.Column(scale=1):
                refresh_codebook_btn = gr.Button("🔄 Refresh Categories", variant="secondary")
                refresh_stats_btn = gr.Button("Refresh Statistics", variant="primary")
        
        with gr.Row():
            metrics_display = RichTextbox(
                label="Metrics",
                interactive=False
            )
        with gr.Row():
            confusion_matrix_plot = gr.Plot(
                label="Confusion Matrix"
            )
            
        with gr.Row():
            gr.Markdown("## Disagreements")
            
        with gr.Row():
            with gr.Column():
                disagreement_index_slider = gr.Slider(
                    minimum=0,
                    maximum=100,  # Default value, will be updated
                    step=1,
                    value=0,
                    label="Navigate Disagreements",
                    interactive=True
                )
                
        with gr.Row():
            with gr.Column():
                model_annotation = RichTextbox(
                label="Model Annotation",
                interactive=False
            )
            with gr.Column():
                human_annotation = RichTextbox(
                    label="Human Annotation",
                    interactive=False
                )

        with gr.Row():
            disagreement_index_display = gr.Markdown("**Current Disagreement:** 0 of 0")
            
        with gr.Row():
            disagreement_text = RichTextbox(
                label="Text Content",
                interactive=False
            )
            
       

        ############################################################
        # Event handlers
        ############################################################

        def update_statistics(category):
            """Update statistics for selected category"""
            try:
                if not category:
                    return "Please select a category", None, None, None, "", "", ""
                    
                auto_col = f"autofill_{category}"
                user_col = f"user_{category}"
                
                if auto_col not in annotator.df.columns or user_col not in annotator.df.columns:
                    return "No comparison data available for this category", None, None, None, "", "", ""
                    
                # Get only rows where both auto and user annotations exist
                mask = annotator.df[auto_col].notna() & annotator.df[user_col].notna()
                if not mask.any():
                    return "No matching annotations found for comparison", None, None, None, "", "", ""
                    
                # Get the attribute type and category icons from codebook
                codebook = annotator.load_codebook()
                attr_type = 'categorical'  # default
                category_icons = {}  # Dictionary to store category-icon mappings
                for code in codebook:
                    if code['attribute'] == category:
                        attr_type = code.get('type', 'categorical')
                        # Create mapping of categories to their icons
                        for cat in code.get('categories', []):
                            category_icons[cat['category']] = cat.get('icon', '')
                        break

                y_true = annotator.df[user_col][mask]
                y_pred = annotator.df[auto_col][mask]

                # Create disagreements table
                disagreements_df = None
                if len(y_true) > 0:
                    # Get indices where annotations differ
                    diff_mask = y_true != y_pred
                    if diff_mask.any():
                        # Create DataFrame with relevant columns and add icons
                        disagreements_df = pd.DataFrame({
                            'Original_Index': annotator.df.index[mask][diff_mask],
                            'Text': annotator.df['text'][mask][diff_mask],
                            'Model Annotation': y_pred[diff_mask].apply(lambda x: f"{category_icons.get(x, '')} {x}" if attr_type == 'categorical' else x),
                            'Human Annotation': y_true[diff_mask].apply(lambda x: f"{category_icons.get(x, '')} {x}" if attr_type == 'categorical' else x)
                        })

                # Store disagreements in annotator for navigation
                annotator.current_disagreements = disagreements_df if disagreements_df is not None else pd.DataFrame()
                
                # Update slider maximum if there are disagreements
                slider_props = {
                    "maximum": len(annotator.current_disagreements) - 1 if disagreements_df is not None else 0,
                    "value": 0
                }
                
                # Initialize disagreement view if there are disagreements
                if disagreements_df is not None and len(disagreements_df) > 0:
                    first_row = disagreements_df.iloc[0]
                    disagreement_text = first_row['Text']
                    model_annot = str(first_row['Model Annotation'])
                    human_annot = str(first_row['Human Annotation'])
                    current_display = f"**Current Disagreement:** 1 of {len(disagreements_df)}"
                else:
                    disagreement_text = "No disagreements found"
                    model_annot = ""
                    human_annot = ""
                    current_display = "**Current Disagreement:** 0 of 0"

                if attr_type == 'freetext':
                    try:
                        # BERTScore calculation
                        scorer = BERTScorer(
                            model_type="bert-base-uncased",
                            num_layers=None
                        )
                        
                        # Calculate BERTScore
                        P, R, F1 = scorer.score(y_pred.tolist(), y_true.tolist())
                        
                        # Calculate mean scores
                        mean_bert_p = P.mean().item()
                        mean_bert_r = R.mean().item()
                        mean_bert_f1 = F1.mean().item()

                        # Calculate ROUGE scores
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                        
                        # Calculate ROUGE scores and store in lists for plotting
                        rouge_scores = {'rouge1': {'p': [], 'r': [], 'f': []},
                                      'rouge2': {'p': [], 'r': [], 'f': []},
                                      'rougeL': {'p': [], 'r': [], 'f': []}}
                        
                        for ref, pred in zip(y_true, y_pred):
                            scores = scorer.score(ref, pred)
                            for metric in ['rouge1', 'rouge2', 'rougeL']:
                                rouge_scores[metric]['p'].append(scores[metric].precision)
                                rouge_scores[metric]['r'].append(scores[metric].recall)
                                rouge_scores[metric]['f'].append(scores[metric].fmeasure)
                        
                        # Calculate averages
                        avg_scores = {metric: {
                            'p': sum(values['p'])/len(values['p']),
                            'r': sum(values['r'])/len(values['r']),
                            'f': sum(values['f'])/len(values['f'])
                        } for metric, values in rouge_scores.items()}
                        
                        # Create visualization
                        plt.close('all')
                        plt.style.use('dark_background')
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Plot 1: BERTScore distributions
                        sns.kdeplot(data=P, label='Precision', ax=ax1, color='blue')
                        sns.kdeplot(data=R, label='Recall', ax=ax1, color='green')
                        sns.kdeplot(data=F1, label='F1', ax=ax1, color='red')
                        ax1.axvline(mean_bert_p, color='blue', linestyle='--', alpha=0.5)
                        ax1.axvline(mean_bert_r, color='green', linestyle='--', alpha=0.5)
                        ax1.axvline(mean_bert_f1, color='red', linestyle='--', alpha=0.5)
                        ax1.set_title('BERTScore Distribution', color='white', pad=20)
                        ax1.set_xlabel('Score', color='white')
                        ax1.set_ylabel('Density', color='white')
                        ax1.tick_params(colors='white')
                        ax1.legend()

                        # Plot 2: ROUGE F1 distributions
                        sns.kdeplot(data=rouge_scores['rouge1']['f'], label='ROUGE-1', ax=ax2, color='blue')
                        sns.kdeplot(data=rouge_scores['rouge2']['f'], label='ROUGE-2', ax=ax2, color='green')
                        sns.kdeplot(data=rouge_scores['rougeL']['f'], label='ROUGE-L', ax=ax2, color='red')
                        ax2.set_title('ROUGE Score Distribution', color='white', pad=20)
                        ax2.set_xlabel('F1 Score', color='white')
                        ax2.set_ylabel('Density', color='white')
                        ax2.tick_params(colors='white')
                        ax2.legend()

                        plt.tight_layout()
                        
                        # Create metrics text
                        metrics = [
                            f"[b][u]BERTScore Metrics[/u][/b]",
                            f"Precision: {mean_bert_p:.3f}",
                            f"Recall: {mean_bert_r:.3f}",
                            f"F1: {mean_bert_f1:.3f}",
                            f"",
                            f"[b][u]ROUGE-1 Metrics[/u][/b]",
                            f"Precision: {avg_scores['rouge1']['p']:.3f}",
                            f"Recall: {avg_scores['rouge1']['r']:.3f}",
                            f"F1: {avg_scores['rouge1']['f']:.3f}",
                            f"",
                            f"[b][u]ROUGE-2 Metrics[/u][/b]",
                            f"Precision: {avg_scores['rouge2']['p']:.3f}",
                            f"Recall: {avg_scores['rouge2']['r']:.3f}",
                            f"F1: {avg_scores['rouge2']['f']:.3f}",
                            f"",
                            f"[b][u]ROUGE-L Metrics[/u][/b]",
                            f"Precision: {avg_scores['rougeL']['p']:.3f}",
                            f"Recall: {avg_scores['rougeL']['r']:.3f}",
                            f"F1: {avg_scores['rougeL']['f']:.3f}",
                            f"",
                            f"[b][u]Samples Compared[/u][/b]: {len(y_true)}",
                        ]
                        
                        metrics_text = "<br>".join(metrics)
                        return (
                            metrics_text, 
                            fig, 
                            gr.Slider(**slider_props),
                            current_display,
                            disagreement_text,
                            model_annot,
                            human_annot
                        )
                        
                    except Exception as e:
                        print(f"Error calculating text similarity metrics: {e}")
                        return f"Error calculating text similarity metrics: {str(e)}", None, None, None, "", "", ""
                
                else:
                    # Original categorical metrics and confusion matrix code
                    metrics = [
                        f"[b][u]Agreement Rate[/u][/b]: {(y_true == y_pred).mean():.3f}",
                        f"[b][u]Accuracy[/u][/b]: {accuracy_score(y_true, y_pred):.3f}",
                        f"[b][u]Macro F1[/u][/b]: {f1_score(y_true, y_pred, average='macro'):.3f}",
                        f"[b][u]Weighted F1[/u][/b]: {f1_score(y_true, y_pred, average='weighted'):.3f}",
                        f"[b][u]Samples Compared[/u][/b]: {len(y_true)}"
                    ]

                    metrics_text = "<br><br>".join(metrics)
                    
                    plt.close('all')
                    plt.style.use('dark_background')
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    labels = sorted(list(set(y_true) | set(y_pred)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d',
                        cmap=sns.dark_palette("#69d", as_cmap=True),
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax,
                        cbar_kws={'label': 'Count'},
                        annot_kws={'color': 'white', 'fontsize': 10}
                    )
                    
                    plt.title(f'Confusion Matrix - {category}', color='white', pad=20)
                    plt.ylabel('Human annotation', color='white')
                    plt.xlabel('Model annotation', color='white')
                    ax.tick_params(colors='white')
                    plt.tight_layout()
                    
                    return (
                        metrics_text, 
                        fig, 
                        gr.Slider(**slider_props),
                        current_display,
                        disagreement_text,
                        model_annot,
                        human_annot
                    )
                    
            except Exception as e:
                print(f"Error in update_statistics: {str(e)}")
                return f"Error calculating statistics: {str(e)}", None, None, None, "", "", ""

        def refresh_status_categories():
            """Updates the category dropdown in the Status tab"""
            try:
                categories = [code["attribute"] for code in annotator.load_codebook()]
                return gr.Dropdown(choices=categories)
            except Exception as e:
                print(f"Error refreshing status categories: {e}")
                return gr.Dropdown(choices=[])

        def navigate_disagreement(index):
            """Navigate through disagreements using slider"""
            try:
                if not hasattr(annotator, 'current_disagreements') or annotator.current_disagreements.empty:
                    return "No disagreements to display", "", "", "**Current Disagreement:** 0 of 0"
                
                row = annotator.current_disagreements.iloc[int(index)]
                return (
                    row['Text'],
                    str(row['Model Annotation']),
                    str(row['Human Annotation']),
                    f"**Current Disagreement:** {int(index) + 1} of {len(annotator.current_disagreements)} (Original Index: {row['Original_Index']})"
                )
            except Exception as e:
                print(f"Error navigating disagreements: {e}")
                return "Error displaying disagreement", "", "", "**Current Disagreement:** 0 of 0"

        # Connect event handlers
        refresh_stats_btn.click(
            fn=update_statistics,
            inputs=[category_select],
            outputs=[
                metrics_display,
                confusion_matrix_plot,
                disagreement_index_slider,
                disagreement_index_display,
                disagreement_text,
                model_annotation,
                human_annotation
            ]
        )

        refresh_codebook_btn.click(
            fn=refresh_status_categories,
            outputs=[category_select]
        )

        # Add new event handler for disagreement navigation
        disagreement_index_slider.change(
            fn=navigate_disagreement,
            inputs=[disagreement_index_slider],
            outputs=[
                disagreement_text,
                model_annotation,
                human_annotation,
                disagreement_index_display
            ]
        )

        # Only attach the load event if demo is provided
        if demo is not None:
            demo.load(
                fn=get_review_progress,
                outputs=[review_progress]
            )

        return {
            'category_select': category_select,
            'metrics_display': metrics_display,
            'confusion_matrix_plot': confusion_matrix_plot,
            'disagreement_index_slider': disagreement_index_slider,
            'disagreement_text': disagreement_text,
            'model_annotation': model_annotation,
            'human_annotation': human_annotation
        } 