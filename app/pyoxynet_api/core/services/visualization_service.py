"""
Visualization Service for CPET Analysis
Handles all plotting and chart generation for CPET data visualization
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any


class CPETVisualizationService:
    """
    Service for generating CPET analysis visualizations
    
    Handles:
    - Core CPET plots (VO2 vs time, VCO2 vs VO2, etc.)
    - Exercise domain probability plots
    - Metabolic analysis visualizations
    - Interactive Plotly charts with consistent theming
    """
    
    # Color schemes for consistent theming
    DOMAIN_COLORS = {
        'Moderate': '#2E86AB',  # Blue
        'Heavy': '#A23B72',     # Purple
        'Severe': '#F18F01'     # Orange
    }
    
    VARIABLE_COLORS = {
        'VO2': '#1f77b4',
        'VCO2': '#ff7f0e', 
        'VE': '#2ca02c',
        'HR': '#d62728',
        'RER': '#9467bd'
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_vo2_time_plot(self, data: pd.DataFrame, ml_results: Optional[Dict] = None) -> Dict:
        """
        Create VO2 vs time plot with domain probabilities
        
        Args:
            data: CPET data DataFrame
            ml_results: ML analysis results with domain predictions
            
        Returns:
            Plotly figure as dictionary
        """
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=['VO2 vs Time', 'Exercise Domain Probabilities'],
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Main VO2 plot
            time_col = self._get_time_column(data)
            
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data['VO2'],
                    mode='lines',
                    name='VO2',
                    line=dict(color=self.VARIABLE_COLORS['VO2'], width=2),
                    hovertemplate='Time: %{x}<br>VO2: %{y} ml/min<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add VT markers if available
            if ml_results and 'ventilatory_thresholds' in ml_results:
                vt1 = ml_results['ventilatory_thresholds'].get('VT1')
                vt2 = ml_results['ventilatory_thresholds'].get('VT2')
                
                if vt1:
                    fig.add_hline(y=vt1, line_dash="dash", line_color="red",
                                annotation_text="VT1", row=1, col=1)
                if vt2:
                    fig.add_hline(y=vt2, line_dash="dash", line_color="orange",
                                annotation_text="VT2", row=1, col=1)
            
            # Domain probabilities subplot
            if ml_results and 'temporal_analysis' in ml_results:
                predictions = np.array(ml_results['temporal_analysis']['predictions_over_time'])
                time_points = np.linspace(data[time_col].min(), data[time_col].max(), len(predictions))
                
                for i, domain in enumerate(['Moderate', 'Heavy', 'Severe']):
                    fig.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=predictions[:, i],
                            mode='lines',
                            name=domain,
                            line=dict(color=self.DOMAIN_COLORS[domain]),
                            fill='tonexty' if i > 0 else None,
                            hovertemplate=f'{domain}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="VO2 (ml/min)", row=1, col=1)
            fig.update_yaxes(title_text="Probability", row=2, col=1)
            
            fig.update_layout(
                title="CPET Analysis: VO2 vs Time with Exercise Domains",
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            return {'figure': fig.to_dict(), 'success': True, 'error': None}
            
        except Exception as e:
            self.logger.error(f"VO2 time plot creation failed: {e}")
            return {'figure': None, 'success': False, 'error': str(e)}
    
    def create_vo2_vco2_plot(self, data: pd.DataFrame, ml_results: Optional[Dict] = None) -> Dict:
        """Create VO2 vs VCO2 plot (V-slope method)"""
        try:
            fig = go.Figure()
            
            # Main scatter plot
            fig.add_trace(
                go.Scatter(
                    x=data['VO2'],
                    y=data['VCO2'],
                    mode='markers+lines',
                    name='VCO2 vs VO2',
                    line=dict(color=self.VARIABLE_COLORS['VCO2']),
                    marker=dict(size=4),
                    hovertemplate='VO2: %{x} ml/min<br>VCO2: %{y} ml/min<br>RER: %{customdata:.3f}<extra></extra>',
                    customdata=data['VCO2']/data['VO2'] if 'VCO2' in data.columns else None
                )
            )
            
            # Add RER = 1.0 reference line
            vo2_range = [data['VO2'].min(), data['VO2'].max()]
            fig.add_trace(
                go.Scatter(
                    x=vo2_range,
                    y=vo2_range,
                    mode='lines',
                    name='RER = 1.0',
                    line=dict(color='red', dash='dash', width=1),
                    hovertemplate='RER = 1.0 reference<extra></extra>'
                )
            )
            
            # Add VT markers if available
            if ml_results and 'ventilatory_thresholds' in ml_results:
                vt1 = ml_results['ventilatory_thresholds'].get('VT1')
                vt2 = ml_results['ventilatory_thresholds'].get('VT2')
                
                if vt1 and vt1 in data['VO2'].values:
                    vt1_vco2 = data[data['VO2'] == vt1]['VCO2'].iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[vt1], y=[vt1_vco2],
                            mode='markers',
                            name='VT1',
                            marker=dict(color='red', size=10, symbol='diamond')
                        )
                    )
                
                if vt2 and vt2 in data['VO2'].values:
                    vt2_vco2 = data[data['VO2'] == vt2]['VCO2'].iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[vt2], y=[vt2_vco2],
                            mode='markers',
                            name='VT2',
                            marker=dict(color='orange', size=10, symbol='diamond')
                        )
                    )
            
            fig.update_layout(
                title="VCO2 vs VO2 (V-slope Method)",
                xaxis_title="VO2 (ml/min)",
                yaxis_title="VCO2 (ml/min)",
                showlegend=True,
                hovermode='closest'
            )
            
            return {'figure': fig.to_dict(), 'success': True, 'error': None}
            
        except Exception as e:
            self.logger.error(f"VO2 vs VCO2 plot creation failed: {e}")
            return {'figure': None, 'success': False, 'error': str(e)}
    
    def create_ventilatory_equivalents_plot(self, data: pd.DataFrame) -> Dict:
        """Create ventilatory equivalents plot (VE/VO2 and VE/VCO2 vs time)"""
        try:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=['Ventilatory Equivalents vs Time']
            )
            
            time_col = self._get_time_column(data)
            
            # Calculate ventilatory equivalents
            ve_vo2 = (data['VE'] * 1000) / data['VO2'] if 'VE' in data.columns else None
            ve_vco2 = (data['VE'] * 1000) / data['VCO2'] if 'VE' in data.columns else None
            
            if ve_vo2 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=data[time_col],
                        y=ve_vo2,
                        mode='lines',
                        name='VE/VO2',
                        line=dict(color='blue', width=2),
                        hovertemplate='Time: %{x}s<br>VE/VO2: %{y:.1f}<extra></extra>'
                    )
                )
            
            if ve_vco2 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=data[time_col],
                        y=ve_vco2,
                        mode='lines',
                        name='VE/VCO2',
                        line=dict(color='orange', width=2),
                        hovertemplate='Time: %{x}s<br>VE/VCO2: %{y:.1f}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title="Ventilatory Equivalents vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Ventilatory Equivalents",
                showlegend=True,
                hovermode='x unified'
            )
            
            return {'figure': fig.to_dict(), 'success': True, 'error': None}
            
        except Exception as e:
            self.logger.error(f"Ventilatory equivalents plot creation failed: {e}")
            return {'figure': None, 'success': False, 'error': str(e)}
    
    def create_nine_panel_plot(self, data: pd.DataFrame, ml_results: Optional[Dict] = None) -> Dict:
        """Create comprehensive 9-panel CPET analysis plot"""
        try:
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    'VO2 vs Time', 'VCO2 vs Time', 'VE vs Time',
                    'HR vs Time', 'RER vs Time', 'VCO2 vs VO2',
                    'VE/VO2 vs Time', 'VE/VCO2 vs Time', 'Domain Probabilities'
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            time_col = self._get_time_column(data)
            
            # Row 1: Primary variables
            variables_row1 = [
                ('VO2', self.VARIABLE_COLORS['VO2'], 'ml/min'),
                ('VCO2', self.VARIABLE_COLORS['VCO2'], 'ml/min'),
                ('VE', self.VARIABLE_COLORS['VE'], 'L/min')
            ]
            
            for i, (var, color, unit) in enumerate(variables_row1, 1):
                if var in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data[time_col], y=data[var],
                            mode='lines', name=var,
                            line=dict(color=color, width=1.5),
                            showlegend=False,
                            hovertemplate=f'%{{y:.0f}} {unit}<extra></extra>'
                        ),
                        row=1, col=i
                    )
            
            # Row 2: Secondary variables
            if 'HR' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data[time_col], y=data['HR'],
                        mode='lines', name='HR',
                        line=dict(color=self.VARIABLE_COLORS['HR'], width=1.5),
                        showlegend=False,
                        hovertemplate='%{y:.0f} bpm<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # RER
            if 'RER' in data.columns or ('VO2' in data.columns and 'VCO2' in data.columns):
                rer_data = data['RER'] if 'RER' in data.columns else data['VCO2'] / data['VO2']
                fig.add_trace(
                    go.Scatter(
                        x=data[time_col], y=rer_data,
                        mode='lines', name='RER',
                        line=dict(color=self.VARIABLE_COLORS['RER'], width=1.5),
                        showlegend=False,
                        hovertemplate='%{y:.3f}<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # VCO2 vs VO2
            fig.add_trace(
                go.Scatter(
                    x=data['VO2'], y=data['VCO2'],
                    mode='lines', name='VCO2 vs VO2',
                    line=dict(color='purple', width=1.5),
                    showlegend=False,
                    hovertemplate='VO2: %{x}<br>VCO2: %{y}<extra></extra>'
                ),
                row=2, col=3
            )
            
            # Row 3: Derived variables
            # VE/VO2
            if 'VE' in data.columns:
                ve_vo2 = (data['VE'] * 1000) / data['VO2']
                fig.add_trace(
                    go.Scatter(
                        x=data[time_col], y=ve_vo2,
                        mode='lines', name='VE/VO2',
                        line=dict(color='green', width=1.5),
                        showlegend=False,
                        hovertemplate='%{y:.1f}<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # VE/VCO2
                ve_vco2 = (data['VE'] * 1000) / data['VCO2']
                fig.add_trace(
                    go.Scatter(
                        x=data[time_col], y=ve_vco2,
                        mode='lines', name='VE/VCO2',
                        line=dict(color='brown', width=1.5),
                        showlegend=False,
                        hovertemplate='%{y:.1f}<extra></extra>'
                    ),
                    row=3, col=2
                )
            
            # Domain probabilities
            if ml_results and 'temporal_analysis' in ml_results:
                predictions = np.array(ml_results['temporal_analysis']['predictions_over_time'])
                time_points = np.linspace(data[time_col].min(), data[time_col].max(), len(predictions))
                
                for i, (domain, color) in enumerate(self.DOMAIN_COLORS.items()):
                    fig.add_trace(
                        go.Scatter(
                            x=time_points, y=predictions[:, i],
                            mode='lines', name=domain,
                            line=dict(color=color, width=1.5),
                            showlegend=i == 0,  # Only show legend for first trace
                            hovertemplate=f'{domain}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=3, col=3
                    )
            
            # Update layout
            fig.update_layout(
                title="Comprehensive CPET Analysis (9-Panel View)",
                height=800,
                showlegend=True,
                hovermode='closest'
            )
            
            # Update axes titles
            axes_titles = [
                ('Time (s)', 'VO2 (ml/min)'), ('Time (s)', 'VCO2 (ml/min)'), ('Time (s)', 'VE (L/min)'),
                ('Time (s)', 'HR (bpm)'), ('Time (s)', 'RER'), ('VO2 (ml/min)', 'VCO2 (ml/min)'),
                ('Time (s)', 'VE/VO2'), ('Time (s)', 'VE/VCO2'), ('Time (s)', 'Probability')
            ]
            
            for i, (xlabel, ylabel) in enumerate(axes_titles, 1):
                row = (i - 1) // 3 + 1
                col = (i - 1) % 3 + 1
                fig.update_xaxes(title_text=xlabel, row=row, col=col)
                fig.update_yaxes(title_text=ylabel, row=row, col=col)
            
            return {'figure': fig.to_dict(), 'success': True, 'error': None}
            
        except Exception as e:
            self.logger.error(f"Nine-panel plot creation failed: {e}")
            return {'figure': None, 'success': False, 'error': str(e)}
    
    def create_domain_summary_plot(self, ml_results: Dict) -> Dict:
        """Create summary plot of exercise domain analysis"""
        try:
            if 'domain_probabilities' not in ml_results:
                raise ValueError("No domain probabilities in ML results")
            
            domains = list(ml_results['domain_probabilities'].keys())
            probabilities = list(ml_results['domain_probabilities'].values())
            colors = [self.DOMAIN_COLORS[domain] for domain in domains]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=domains,
                    y=probabilities,
                    marker_color=colors,
                    text=[f'{p:.1%}' for p in probabilities],
                    textposition='auto',
                    hovertemplate='%{x}: %{y:.1%}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Exercise Domain Classification Summary",
                xaxis_title="Exercise Domain",
                yaxis_title="Probability",
                yaxis=dict(tickformat='.0%', range=[0, 1]),
                showlegend=False
            )
            
            # Add confidence indicator
            confidence = ml_results.get('confidence', 0)
            fig.add_annotation(
                text=f"Confidence: {confidence:.1%}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
            
            return {'figure': fig.to_dict(), 'success': True, 'error': None}
            
        except Exception as e:
            self.logger.error(f"Domain summary plot creation failed: {e}")
            return {'figure': None, 'success': False, 'error': str(e)}
    
    def _get_time_column(self, data: pd.DataFrame) -> str:
        """Get the appropriate time column from data"""
        time_columns = ['TIME', 'time', 'Time', 't', 'T']
        for col in time_columns:
            if col in data.columns:
                return col
        
        # If no time column found, create one from index
        if 'Index' not in data.columns:
            data = data.copy()
            data['Index'] = range(len(data))
        return 'Index'
    
    def get_plot_config(self) -> Dict:
        """Get standard plot configuration for all charts"""
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'cpet_analysis',
                'height': 600,
                'width': 800,
                'scale': 2
            }
        }