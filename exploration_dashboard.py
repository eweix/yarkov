import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)
n_samples = 300
n_generations = 20

k_syn = np.random.uniform(0.1, 1.0, n_samples)
M_crit = np.random.uniform(10, 100, n_samples)
a = np.random.uniform(0.1, 2.0, n_samples)

data = []
for i in range(n_samples):
    base_mean = 100 + 50 * k_syn[i] + 0.5 * M_crit[i]
    base_var = 20 + 30 * a[i]
    for g in range(n_generations):
        data.append({
            'sample_id': i,
            'generation': g,
            'k_syn': k_syn[i],
            'M_crit': M_crit[i],
            'a': a[i],
            'mean': base_mean + 10 * g + np.random.randn() * 5,
            'var': base_var + 2 * g + np.random.randn() * 3,
            'skew': np.random.uniform(-0.5, 0.5),
            'kurt': 3 + np.random.uniform(-0.5, 0.5)
        })

df = pd.DataFrame(data)


def create_dashboard(df, output_file='exploration_dashboard.html'):
    moments = ['mean', 'var']
    colors = ['#1f77b4', '#ff7f0e']
    
    n_samples = df['sample_id'].nunique()
    n_gens = df['generation'].max() + 1
    
    unique_samples = df.groupby('sample_id').first().reset_index()
    k_syn_vals = unique_samples['k_syn'].values
    M_crit_vals = unique_samples['M_crit'].values
    a_vals = unique_samples['a'].values
    
    a_min, a_max = a_vals.min(), a_vals.max()
    a_scaled = 5 + 10 * (a_vals - a_min) / (a_max - a_min)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time Series (all 300 runs)', 'Parameter Space (k_syn vs M_crit)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.08
    )
    
    for sid in range(n_samples):
        sample_df = df[df['sample_id'] == sid].sort_values('generation')
        for mi, moment in enumerate(moments):
            vis = mi == 0
            fig.add_trace(
                go.Scatter(
                    x=sample_df['generation'].values,
                    y=sample_df[moment].values,
                    mode='lines',
                    line=dict(color=colors[mi], width=1),
                    opacity=0.35,
                    hoverinfo='text',
                    text=[f'sid={sid}<br>k_syn={k_syn_vals[sid]:.3f}<br>M_crit={M_crit_vals[sid]:.1f}<br>a={a_vals[sid]:.2f}<br>{moment}={v:.1f}'
                          for v in sample_df[moment].values],
                    name=f'{moment}_{sid}',
                    showlegend=False,
                    visible=vis,
                    legendgroup=moment
                ),
                row=1, col=1
            )
    
    for mi, moment in enumerate(moments):
        gen_df = df[df['generation'] == 0].sort_values('sample_id')
        fig.add_trace(
            go.Scatter(
                x=gen_df['k_syn'].values,
                y=gen_df['M_crit'].values,
                mode='markers',
                marker=dict(
                    size=a_scaled,
                    color=gen_df[moment].values,
                    colorscale='RdBu_r',
                    opacity=0.6,
                    colorbar=dict(title=moment, x=1.02, len=0.5, y=0.75 if mi == 0 else 0.25)
                ),
                customdata=np.column_stack([
                    gen_df['sample_id'].values,
                    gen_df['k_syn'].values,
                    gen_df['M_crit'].values,
                    gen_df['a'].values
                ]),
                hovertemplate='<b>sample_id: %{customdata[0]}</b><br>' +
                            'k_syn: %{customdata[1]:.3f}<br>' +
                            'M_crit: %{customdata[2]:.1f}<br>' +
                            'a: %{customdata[3]:.2f}<br>' +
                            f'{moment}: %{{marker.color:.1f}}<extra></extra>',
                name=f'param_{moment}',
                showlegend=False,
                visible=(mi == 0),
                legendgroup=moment
            ),
            row=1, col=2
        )
    
    def build_visibility(moment_idx, gen):
        vis = []
        for sid in range(n_samples):
            for mi, _ in enumerate(moments):
                vis.append(mi == moment_idx)
        for mi, _ in enumerate(moments):
            vis.append(mi == moment_idx and gen == 0)
        return vis
    
    slider_steps = []
    for g in range(n_gens):
        args_list = [{'visible': build_visibility(0, g)}]
        step_dict = {'args': args_list}
        step_dict['label'] = str(g)
        slider_steps.append(step_dict)
    
    fig.update_layout(
        title=dict(text='Protein Expression: Mean Evolution', font=dict(size=16)),
        height=550, width=1150,
        template='plotly_white',
        xaxis=dict(title='Generation'),
        yaxis=dict(title='Mean'),
        xaxis2=dict(title='k_syn'),
        yaxis2=dict(title='M_crit'),
        updatemenus=[dict(
            type='dropdown',
            x=0.0, y=1.12,
            buttons=[
                dict(label='Mean', method='update',
                     args=[{'visible': build_visibility(0, 0)},
                           {'title.text': 'Protein Expression: Mean Evolution',
                            'yaxis.title.text': 'Mean'}]),
                dict(label='Variance', method='update',
                     args=[{'visible': build_visibility(1, 0)},
                           {'title.text': 'Protein Expression: Variance Evolution',
                            'yaxis.title.text': 'Variance'}]),
            ],
            showactive=True,
            font=dict(size=11)
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Generation: '),
            pad=dict(t=45),
            steps=slider_steps,
            x=0.1, len=0.8, xanchor='left', y=-0.08, yanchor='top'
        )]
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn', full_html=True)
    print(f"Dashboard saved to {output_file}")


if __name__ == '__main__':
    create_dashboard(df)