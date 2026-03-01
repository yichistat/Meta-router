#!/usr/bin/env python3
"""
Combined R-learner and DR-learner Experiment with Normalized Efficiency
For each ratio point and each approach, calculate total efficiency median, then normalize
Formula: (approach_median - random_median) / (oracle_median - random_median)
"""

import pandas as pd
import numpy as np
import ast
import sklearn
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import warnings
import random, numpy as np
from datetime import datetime
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

warnings.filterwarnings('ignore')
# Set clean style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (8, 8),
    'lines.linewidth': 3,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_and_split_preference_data_with_seed(seed):
    """Load 1_PC50_otstd.csv and split according to requirements with specific seed"""
    data = pd.read_csv("./1_PC50_varstd.csv")
    random.seed(seed)
    np.random.seed(seed)
    
    # Use PC1-PC50 as embeddings (columns embd.PC1 to embd.PC50)
    pc_columns = [f'embd.PC{i}' for i in range(1, 51)]
    embeddings = data[pc_columns].values
    
    print(f"Using PC embeddings with dimensions: {embeddings.shape[1]}")
    print(f"Total samples: {len(data)}")

    # Split data: 500 test, 50 golden, rest preference
    n_samples = len(data)
    indices = np.random.permutation(n_samples)

    # Test split: exactly 500 samples
    test_size = 500
    test_indices = indices[:test_size]

    # Remaining samples for training
    remaining_indices = indices[test_size:]

    # Golden size: exactly 50 samples, rest as preference
    golden_size = 100
    golden_indices = remaining_indices[:golden_size]
    preference_indices = remaining_indices[golden_size:]

    print(f"Data splits - Test: {len(test_indices)}, Golden: {len(golden_indices)}, Preference: {len(preference_indices)}")

    return {
        'golden_train_embeddings': embeddings[golden_indices],
        'golden_train_labels': data.iloc[golden_indices]['GL'].values,
        'preference_train_embeddings': embeddings[preference_indices],
        'preference_train_labels': data.iloc[preference_indices]['PL'].values,
        'preference_train_golden_labels': data.iloc[preference_indices]['GL'].values,
        'test_embeddings': embeddings[test_indices],
        'test_labels': data.iloc[test_indices]['GL'].values
    }

class RLearner(BaseEstimator, RegressorMixin):
    """R-learner implementation for preference data"""

    def __init__(self, outcome_model=None, propensity_model=None, treatment_model=None):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.treatment_model = treatment_model

    def fit(self, X, Y, T):
        # Fit outcome model
        self.outcome_model.fit(X, Y)
        m_pred = self.outcome_model.predict(X)

        # Fit propensity model
        self.propensity_model.fit(X, T)
        if hasattr(self.propensity_model, 'predict_proba'):
            e_pred = self.propensity_model.predict_proba(X)[:, 1]
        else:
            e_pred = self.propensity_model.predict(X)
            e_pred = np.clip(e_pred, 0.1, 0.9)

        # Compute pseudo-outcomes
        numerator = Y - m_pred
        denominator = T - e_pred
        valid_mask = np.abs(denominator) > 0.05
        pseudo_outcomes = np.zeros_like(numerator)
        pseudo_outcomes[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        # Fit treatment effect model
        sample_weights = denominator ** 2
        self.treatment_model.fit(X[valid_mask], pseudo_outcomes[valid_mask], sample_weight=sample_weights[valid_mask])
        return self

    def predict(self, X):
        predictions = self.treatment_model.predict(X)
        return np.clip(predictions, -1.2, 1.2)

class DRLearner(BaseEstimator, RegressorMixin):
    """DR-learner implementation for preference data"""

    def __init__(self, outcome_model=None, propensity_model=None, treatment_model=None):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.treatment_model = treatment_model

    def fit(self, X, Y, T):
        # Fit outcome model
        XT = np.column_stack([X, T])
        self.outcome_model.fit(XT, Y)
        XT1 = np.column_stack([X, np.ones(len(T))])
        XT0 = np.column_stack([X, np.zeros(len(T))])
        mu1_pred = self.outcome_model.predict(XT1)
        mu0_pred = self.outcome_model.predict(XT0)
        muT_pred = self.outcome_model.predict(XT)
        
        # Fit propensity model
        self.propensity_model.fit(X, T)
        if hasattr(self.propensity_model, 'predict_proba'):
            e_pred = self.propensity_model.predict_proba(X)[:, 1]
        else:
            e_pred = self.propensity_model.predict(X)
            e_pred = np.clip(e_pred, 0.05, 0.95)

        # Compute pseudo-outcomes
        numerator = T - e_pred
        denominator = (1 - e_pred) * e_pred
        valid_mask = np.abs(denominator) > 0.05
        pseudo_outcomes = (numerator/denominator) * (Y - muT_pred) + mu1_pred - mu0_pred
        
        # Fit treatment effect model
        self.treatment_model.fit(X[valid_mask], pseudo_outcomes[valid_mask])
        return self

    def predict(self, X):
        predictions = self.treatment_model.predict(X)
        return np.clip(predictions, -1.2, 1.2)

def create_random_forest_models(n_samples, n_features):
    """Create optimized Random Forest models"""
    # Outcome model - Optimized Random Forest
    outcome_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )

    # Propensity model - Optimized Random Forest
    propensity_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )

    # Treatment model (regression) - Optimized Random Forest
    treatment_model = RandomForestRegressor(
        n_estimators=200,        # More trees for better performance
        max_depth=30,           # Deeper trees for complex patterns
        min_samples_split=5,    # Avoid overfitting
        min_samples_leaf=2,     # Avoid overfitting
        max_features="sqrt",    # Good balance
        bootstrap=True,         # Use bootstrap sampling
        oob_score=True,         # Out-of-bag scoring
        n_jobs=-1,              # Use all CPU cores
        random_state=42
    )

    return outcome_model, propensity_model, treatment_model

def train_weighted_rlearner(data_splits, lambda_weight=1.0):
    """Train R-learner with preference data using λ weighting for η(q) training"""
    golden_X = data_splits['golden_train_embeddings']
    golden_y = data_splits['golden_train_labels']
    preference_X = data_splits['preference_train_embeddings']
    preference_y = data_splits['preference_train_labels']

    X = np.vstack([golden_X, preference_X])
    Y = np.hstack([golden_y, preference_y])
    T = np.hstack([np.ones(len(golden_X)), np.zeros(len(preference_X))])  # 1=golden, 0=preference

    n_samples, n_features = X.shape
    outcome_model, propensity_model, treatment_model = create_random_forest_models(n_samples, n_features)

    # Fit R-learner
    rlearner = RLearner(outcome_model, propensity_model, treatment_model)
    rlearner.fit(X, Y, T.astype(int))

    # Learn η(q) - corrected preference labels
    delta_pred = rlearner.predict(X)
    n_golden = len(golden_X)
    delta_preference = delta_pred[n_golden:]
    corrected_preference_y = preference_y + delta_preference

    combined_X = np.vstack([golden_X, preference_X])
    combined_y = np.hstack([golden_y, corrected_preference_y])

    eta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )
    eta_model.fit(combined_X, combined_y)
    return eta_model, None

def train_weighted_drlearner(data_splits, lambda_weight=1.0):
    """Train DR-learner with preference data using λ weighting for η(q) training"""
    golden_X = data_splits['golden_train_embeddings']
    golden_y = data_splits['golden_train_labels']
    preference_X = data_splits['preference_train_embeddings']
    preference_y = data_splits['preference_train_labels']

    X = np.vstack([golden_X, preference_X])
    Y = np.hstack([golden_y, preference_y])
    T = np.hstack([np.ones(len(golden_X)), np.zeros(len(preference_X))])  # 1=golden, 0=preference

    n_samples, n_features = X.shape
    outcome_model, propensity_model, treatment_model = create_random_forest_models(n_samples, n_features)

    # Fit DR-learner
    drlearner = DRLearner(outcome_model, propensity_model, treatment_model)
    drlearner.fit(X, Y, T.astype(int))

    # Learn η(q) - corrected preference labels
    delta_pred = drlearner.predict(X)
    n_golden = len(golden_X)
    delta_preference = delta_pred[n_golden:]
    corrected_preference_y = preference_y + delta_preference

    combined_X = np.vstack([golden_X, preference_X])
    combined_y = np.hstack([golden_y, corrected_preference_y])

    eta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )
    eta_model.fit(combined_X, combined_y)
    return eta_model, None

def train_golden_only_random_forest(data_splits):
    """Train optimized Random Forest using only golden data"""
    golden_X = data_splits['golden_train_embeddings']
    golden_y = data_splits['golden_train_labels']

    eta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )
    eta_model.fit(golden_X, golden_y)

    return eta_model, None

def train_pooled_random_forest(data_splits):
    """Train pooled optimized Random Forest (equal weights for golden + preference, no R-learner)"""
    golden_X = data_splits['golden_train_embeddings']
    golden_y = data_splits['golden_train_labels']
    preference_X = data_splits['preference_train_embeddings']
    preference_y = data_splits['preference_train_labels']

    # Combine all data with equal weights
    combined_X = np.vstack([golden_X, preference_X])
    combined_y = np.hstack([golden_y, preference_y])

    eta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )
    eta_model.fit(combined_X, combined_y)

    return eta_model, None

def train_all_golden_random_forest(data_splits):
    """Train optimized Random Forest using golden labels for ALL training data (including preference data)"""
    golden_X = data_splits['golden_train_embeddings']
    golden_y = data_splits['golden_train_labels']
    preference_X = data_splits['preference_train_embeddings']
    preference_golden_y = data_splits['preference_train_golden_labels']  # Use golden labels for preference data too

    # Combine all data using golden labels for everything
    combined_X = np.vstack([golden_X, preference_X])
    combined_y = np.hstack([golden_y, preference_golden_y])

    eta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )
    eta_model.fit(combined_X, combined_y)

    return eta_model, None

def train_preference_only_random_forest(data_splits):
    """Train optimized Random Forest using only preference data"""
    preference_X = data_splits['preference_train_embeddings']
    preference_y = data_splits['preference_train_labels']

    eta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        random_state=42
    )
    eta_model.fit(preference_X, preference_y)

    return eta_model, None

def train_random_baseline(data_splits):
    """Random baseline - no actual training needed"""
    return None, None

def evaluate_total_efficiency_vs_ratio(eta_model, test_data, debug_name=""):
    """
    NEW: Return total efficiency (sum of selected golden labels) for different ratios
    """
    test_X = test_data['test_embeddings']
    test_y = test_data['test_labels']
    
    n_test = len(test_y)

    # Handle Random baseline case
    if eta_model is None:
        # Random selection: shuffle and include samples randomly
        random_indices = np.random.permutation(n_test)
        random_labels = test_y[random_indices]
        
        # Create curve from 0 to all samples
        sample_counts = np.arange(0, n_test + 1)
        ratios = sample_counts / n_test
        total_efficiencies = []
        
        for n_include in sample_counts:
            if n_include == 0:
                total_eff = 0.0
            else:
                total_eff = random_labels[:n_include].sum()  # Sum of golden labels
            
            total_efficiencies.append(total_eff)
        
        return ratios, np.array(total_efficiencies)

    # Predict η(q)
    eta_pred = eta_model.predict(test_X)

    # Sort by η(q) values (descending)
    sorted_indices = np.argsort(-eta_pred)
    sorted_eta = eta_pred[sorted_indices]
    sorted_labels = test_y[sorted_indices]

    # Find how many samples have positive η(q)
    positive_mask = sorted_eta > 0
    n_positive = np.sum(positive_mask)
    
    if n_positive == 0:
        # No positive predictions, return curve starting from 0
        ratios = np.linspace(0, 1, 101)
        total_efficiencies = np.zeros_like(ratios)
        return ratios, total_efficiencies
    
    # Create curve: from 0 to include all positive samples
    sample_counts = np.arange(0, n_positive + 1)
    ratios = sample_counts / n_test
    total_efficiencies = []
    
    for n_include in sample_counts:
        if n_include == 0:
            total_eff = 0.0
        else:
            total_eff = sorted_labels[:n_include].sum()  # Sum of golden labels
        
        total_efficiencies.append(total_eff)
    
    return ratios, np.array(total_efficiencies)

def calculate_oracle_efficiency(test_y):
    """Calculate oracle (optimal) total efficiency"""
    # Oracle: select all positive samples (same as All-Golden approach)
    return np.sum(test_y[test_y > 0])

def calculate_efficiency_gain_results(all_results, approach_names):
    """
    NEW: Calculate efficiency gain using the formula:
    efficiency_gain = method_total_efficiency - random_total_efficiency
    """
    print("🔄 Calculating efficiency gain results...")
    
    # Define common ratio points (focus on 0.2 to 0.8 for display)
    common_ratios = np.linspace(0.2, 0.8, 61)
    
    # First, collect total efficiency values for each approach at each ratio
    total_eff_results = {}
    
    for approach in approach_names:
        print(f"   Processing {approach}...")
        
        total_eff_results[approach] = {}
        
        for target_ratio in common_ratios:
            total_effs_at_ratio = []
            
            # Go through each run for this approach
            for run_ratios, run_total_effs in zip(all_results[approach]['ratios'], all_results[approach]['total_efficiencies']):
                if len(run_ratios) > 0 and target_ratio <= max(run_ratios):
                    # Interpolate to find total efficiency at target_ratio
                    total_eff_at_ratio = np.interp(target_ratio, run_ratios, run_total_effs)
                    total_effs_at_ratio.append(total_eff_at_ratio)
            
            # Calculate median for this ratio point
            if total_effs_at_ratio:
                median_total_eff = np.median(total_effs_at_ratio)
                total_eff_results[approach][target_ratio] = median_total_eff
            else:
                total_eff_results[approach][target_ratio] = np.nan
    
    # Now calculate efficiency gain: method - random
    gain_results = {}
    
    for approach in approach_names:
        ratios_list = []
        gain_effs_list = []
        
        for target_ratio in common_ratios:
            approach_median = total_eff_results[approach].get(target_ratio, np.nan)
            random_median = total_eff_results['Random'].get(target_ratio, np.nan)
            
            if not (np.isnan(approach_median) or np.isnan(random_median)):
                # Calculate efficiency gain: method - random
                efficiency_gain = approach_median - random_median
                ratios_list.append(target_ratio)
                gain_effs_list.append(efficiency_gain)
        
        gain_results[approach] = {
            'ratios': np.array(ratios_list),
            'efficiency_gains': np.array(gain_effs_list)
        }
        
        print(f"      → {len(ratios_list)} valid gain points")
    
    return gain_results

def main():
    """Main experiment function"""
    print("🚀 Starting Efficiency Gain Combined Experiment")
    print("📊 Formula: Method Total Efficiency - Random Total Efficiency")
    
    # All approaches to test (excluding Random from final plot)
    approach_names = [
        'R-learner λ=1',    # R-learner with preference weight
        'DR-learner λ=1',   # DR-learner with preference weight
        'Golden-Only',      # Only golden data baseline
        'Pooled Data',      # Equal weight pooled baseline
        'All-Golden',       # All training data using golden labels
        'Preference-Only',  # Only preference data baseline
        'Random'            # Random selection benchmark (for normalization)
    ]

    num_runs = 200

    # Store results for all approaches
    all_results = {}
    for approach in approach_names:
        all_results[approach] = {'ratios': [], 'total_efficiencies': []}
    
    # Store oracle efficiency for each run
    all_results['oracle_efficiency'] = []

    # Run experiment
    for run_idx in range(num_runs):
        print(f"📈 Running experiment {run_idx + 1}/{num_runs}...")

        # Load data with different seed for each run
        data_splits = load_and_split_preference_data_with_seed(42 + run_idx)
        test_data = {
            'test_embeddings': data_splits['test_embeddings'],
            'test_labels': data_splits['test_labels']
        }
        
        # Calculate oracle efficiency for this run
        oracle_eff = calculate_oracle_efficiency(test_data['test_labels'])
        all_results['oracle_efficiency'].append(oracle_eff)

        # Test each approach
        for approach in approach_names:
            if run_idx == 0:
                print(f"   📊 Training {approach}")

            # Train model based on approach
            if approach == 'R-learner λ=1':
                eta_model, _ = train_weighted_rlearner(data_splits, lambda_weight=1.0)
            elif approach == 'DR-learner λ=1':
                eta_model, _ = train_weighted_drlearner(data_splits, lambda_weight=1.0)
            elif approach == 'Golden-Only':
                eta_model, _ = train_golden_only_random_forest(data_splits)
            elif approach == 'Pooled Data':
                eta_model, _ = train_pooled_random_forest(data_splits)
            elif approach == 'All-Golden':
                eta_model, _ = train_all_golden_random_forest(data_splits)
            elif approach == 'Preference-Only':
                eta_model, _ = train_preference_only_random_forest(data_splits)
            elif approach == 'Random':
                eta_model, _ = train_random_baseline(data_splits)

            # Evaluate total efficiency
            ratios, total_effs = evaluate_total_efficiency_vs_ratio(eta_model, test_data, debug_name=approach)

            all_results[approach]['ratios'].append(ratios)
            all_results[approach]['total_efficiencies'].append(total_effs)

    # Calculate efficiency gain results
    gain_results = calculate_efficiency_gain_results(all_results, approach_names)

    # Create plot
    print("🎨 Creating efficiency gain plot...")

    plt.figure(figsize=(8, 8))

    # Colors and styles for approaches (excluding Random since it will be 0)
    plot_approaches = [a for a in approach_names if a != 'Random']
    colors = ['#E74C3C', '#3498DB', '#9B59B6', '#2ECC71', '#F39C12', '#808080']
    linestyles = ['-', '-', '--', '--', '-.', ':']
    markers = ['o', 's', 'v', 'p', 'h', 'D']

    # Plot curves for each approach
    for i, approach in enumerate(plot_approaches):
        if approach in gain_results:
            ratios = gain_results[approach]['ratios']
            gains = gain_results[approach]['efficiency_gains'] / (0.03523782 * 500)

            plt.plot(ratios, gains,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    markevery=max(1, len(ratios)//8),
                    markersize=8,
                    label=approach,
                    alpha=0.8)

    plt.xlabel('Primary Model Usage Ratio')
    plt.ylabel('Efficiency Gain')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.2, 0.8)  # Focus on 0.2 to 0.8
    
    # Determine y-axis limits based on data
    all_gains = []
    for approach in plot_approaches:
        if approach in gain_results:
            all_gains.extend(gain_results[approach]['efficiency_gains'])

    if all_gains:
        y_min = min(min(all_gains) * 1.1 / (0.03523782 * 500), -0.05)
        y_max = max(all_gains) * 1.05 / (0.03523782 * 500)
        plt.ylim(y_min, y_max)

    # Add reference line at y=0 (Random level)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7, label='Random baseline')

    # Add subtle background
    plt.gca().set_facecolor('#FAFAFA')

    # Add caption at bottom

    # Print summary statistics
    print(f"\n" + "="*70)
    print(f"📈 EFFICIENCY GAIN EXPERIMENT RESULTS")
    print(f"📊 Formula: Method Total Efficiency - Random Total Efficiency")
    print(f"📊 Runs: {num_runs}")
    print(f"="*70)

    # Collect results for CSV backup
    results_data = []
    for approach in plot_approaches:
        if approach in gain_results:
            ratios = gain_results[approach]['ratios']
            gains = gain_results[approach]['efficiency_gains']
            max_gain = np.max(gains)
            max_ratio = ratios[np.argmax(gains)]
            auc = np.trapz(gains, ratios)

            print(f"{approach}:")
            print(f"   Max Efficiency Gain: {max_gain:.3f} at {max_ratio:.3f} ratio")
            print(f"   AUC: {auc:.3f}")
            print(f"   Valid points: {len(ratios)}")
            print()
            
            results_data.append({
                'approach': approach,
                'max_efficiency_gain': max_gain,
                'max_ratio': max_ratio,
                'auc': auc,
                'num_runs': num_runs,
                'valid_points': len(ratios),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            })

    # Save individual MC round data before aggregation
    print("💾 Saving individual MC round data...")
    individual_run_data = []
    common_ratios = np.linspace(0.2, 0.8, 61)  # Same ratios used in gain calculation
    
    for run_idx in range(num_runs):
        for approach in approach_names:
            if approach != 'Random':  # Skip Random for efficiency gain calculation
                run_ratios = all_results[approach]['ratios'][run_idx]
                run_total_effs = all_results[approach]['total_efficiencies'][run_idx]
                random_total_effs = all_results['Random']['total_efficiencies'][run_idx]
                random_ratios = all_results['Random']['ratios'][run_idx]
                
                # Calculate efficiency gains for this run
                for target_ratio in common_ratios:
                    if len(run_ratios) > 0 and target_ratio <= max(run_ratios):
                        # Interpolate approach efficiency
                        approach_eff = np.interp(target_ratio, run_ratios, run_total_effs)
                        
                        # Interpolate random efficiency
                        if len(random_ratios) > 0 and target_ratio <= max(random_ratios):
                            random_eff = np.interp(target_ratio, random_ratios, random_total_effs)
                            
                            # Calculate efficiency gain
                            efficiency_gain = approach_eff - random_eff
                            
                            individual_run_data.append({
                                'mc_round': run_idx + 1,
                                'approach': approach,
                                'ratio': target_ratio,
                                'approach_efficiency': approach_eff,
                                'random_efficiency': random_eff,
                                'efficiency_gain': efficiency_gain,
                                'normalized_gain': efficiency_gain / (0.03523782 * 500),
                                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                            })
    
    # Save individual run data to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if individual_run_data:
        individual_df = pd.DataFrame(individual_run_data)
        individual_csv_filename = f'./PC50_varstd_RF_100G_individual_runs.csv'
        individual_df.to_csv(individual_csv_filename, index=False)
        print(f"📁 Individual MC round data saved to: {individual_csv_filename}")
        print(f"📊 Total records: {len(individual_run_data)} (Runs: {num_runs}, Approaches: {len([a for a in approach_names if a != 'Random'])}, Ratios: {len(common_ratios)})")


if __name__ == "__main__":
    main()