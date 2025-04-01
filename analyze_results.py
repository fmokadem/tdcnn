from matplotlib.transforms import Affine2D

# Set style
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 24,
    'figure.figsize': (12, 8),
})

# Updated directories to be specific to the experiment
BASE_EXP_DIR = '/home/fmokadem/NAS/tdcnn/results/exp_20250330_115549'
RESULTS_DIR = BASE_EXP_DIR
LOG_DIR = BASE_EXP_DIR # Log files are in the same directory
OUTPUT_DIR = BASE_EXP_DIR # Base output directory
FIGS_DIR = os.path.join(OUTPUT_DIR, 'figs')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --- Radar Chart Helper Function (Moved to Global Scope) ---
def radar_factory(num_vars, frame='circle'):
    """Creates a RadarAxes projection and returns the angles for the axes."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            # Ensure the plot is closed for radar charts
            self._close_line(lines[0])
            return lines
        def _close_line(self, line):
            x, y = line.get_data()
            # Modifies x, y to close the line if not already closed
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        def _gen_axes_patch(self):
            # The Axes patch must be circular regardless of shape settings.
            return Circle((0.5, 0.5), 0.5)
        def _gen_axes_spines(self):
            # Generate the circular spine path.
            spine = Spine(axes=self, spine_type='circle', path=Path.unit_circle())
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {'polar': spine}
    register_projection(RadarAxes)
    return theta

# --- Data Loading and Baseline Parsing ---
def parse_baseline_from_log(log_file_path):
    """Parses baseline metrics from the corresponding log file."""
    baseline_metrics = {
        'params': None,
        'flops': None,
        'accuracy': None,
        'inference_time': None
    }
    # Regex pattern to find the baseline log line
    # Example: Baseline resnet: params=11173962, FLOPs=413056368.0, accuracy=0.9888, inference_time=0.1506s
    baseline_pattern = re.compile(
        r"Baseline\s+\w+:\s+params=(\d+),?\s+FLOPs=([\d\.e\+\-]+),?\s+accuracy=([\d\.]+),?\s+inference_time=([\d\.]+)s?"
    )
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = baseline_pattern.search(line)
                if match:
                    baseline_metrics['params'] = int(match.group(1))
                    baseline_metrics['flops'] = float(match.group(2))
                    baseline_metrics['accuracy'] = float(match.group(3))
                    baseline_metrics['inference_time'] = float(match.group(4))
                    print(f"  Found baseline metrics in {os.path.basename(log_file_path)}: {baseline_metrics}")
                    return baseline_metrics # Found, no need to read further
    except FileNotFoundError:
        print(f"Error: Log file not found for baseline parsing: {log_file_path}")
    except Exception as e:
        print(f"Error parsing baseline from {log_file_path}: {e}")
        
    print(f"Warning: Baseline metrics not found in {os.path.basename(log_file_path)}")
    return baseline_metrics # Return dictionary with Nones if not found

def load_results_and_baseline():
    """Load the latest results JSON and find corresponding baseline from logs."""
    results_data = {}
    baseline_data = {}
    model_map = {'resnet': 'renet18', 'alexnet': 'alexnet', 'vgg': 'vgg16'} # Map standardized name to log name pattern
    
    # Find the JSON result files first
    json_files = glob.glob(os.path.join(RESULTS_DIR, 'htd-*-*.json'))
    if not json_files:
        print(f"Error: No JSON result files found in {RESULTS_DIR}")
        return {}, {}
        
    processed_models = set()
    latest_json_files = {}

    # Find the latest JSON for each model type
    for f in json_files:
        match = re.search(r"htd-(resnet|alexnet|vgg)-(\d{8}_\d{6})\.json", os.path.basename(f))
        if match:
            model_type = match.group(1)
            timestamp = match.group(2)
            if model_type not in latest_json_files or timestamp > latest_json_files[model_type][0]:
                latest_json_files[model_type] = (timestamp, f)

    # Load the latest JSONs and find corresponding logs for baseline
    for model_type, (timestamp, json_file) in latest_json_files.items():
        print(f"Processing latest JSON for {model_type}: {json_file}")
        try:
            with open(json_file, 'r') as f:
                results_data[model_type] = json.load(f)
                print(f"  Loaded {len(results_data[model_type])} results for {model_type}.")
            
            # Find the corresponding original log file for baseline metrics
            log_name_pattern = model_map.get(model_type)
            if log_name_pattern:
                log_search_pattern = os.path.join(LOG_DIR, f'htd_{log_name_pattern}_*.log')
                log_files = sorted(glob.glob(log_search_pattern))
                if log_files:
                    # Assume the first/oldest matching log file corresponds to the notebook run
                    # Or better, find the one matching the timestamp in the notebook filename if possible?
                    # For now, using the specific log files provided.
                    log_file_map = {
                        'resnet': os.path.join(LOG_DIR, 'htd_renet18_20250329_153202.log'),
                        'alexnet': os.path.join(LOG_DIR, 'htd_alexnet_20250329_151419.log'),
                        'vgg': os.path.join(LOG_DIR, 'htd_vgg16_20250330_082114.log')
                    }
                    log_file_to_parse = log_file_map.get(model_type)
                    if log_file_to_parse and os.path.exists(log_file_to_parse):
                         baseline_data[model_type] = parse_baseline_from_log(log_file_to_parse)
                    else:
                        print(f"Warning: Could not find specified log file for baseline: {log_file_to_parse}")
                        baseline_data[model_type] = parse_baseline_from_log(None) # Will print error and return Nones
                else:
                    print(f"Warning: No log files found matching pattern {log_search_pattern} for baseline.")
                    baseline_data[model_type] = parse_baseline_from_log(None)
            else:
                 print(f"Warning: No log name pattern defined for model type {model_type}.")
                 baseline_data[model_type] = parse_baseline_from_log(None)
                 
        except Exception as e:
            print(f"Error loading or processing {json_file}: {e}")
            
    return results_data, baseline_data

# --- DataFrame Creation ---
def create_summary_df(results, baselines):
    """Create a summary dataframe of the results using provided baselines."""
    summary_data = []
    
    for model_name, model_results in results.items():
        # Get baseline metrics for this model
        baseline = baselines.get(model_name, {})
        baseline_params = baseline.get('params')
        baseline_flops = baseline.get('flops')
        baseline_accuracy = baseline.get('accuracy')
        baseline_inference_time = baseline.get('inference_time')
        
        if None in [baseline_params, baseline_flops, baseline_accuracy, baseline_inference_time]:
            print(f"Warning: Missing baseline metrics for {model_name}. Cannot calculate relative changes.")
            # Optionally, skip this model or handle missing data
            continue 
            
        for i, result in enumerate(model_results):
            # Calculate relative metrics only if baseline exists
            params_reduction = 1 - (result['params'] / baseline_params) if baseline_params else 0
            flops_reduction = 1 - (result['flops'] / baseline_flops) if baseline_flops else 0
            accuracy_change = result['accuracy'] - baseline_accuracy if baseline_accuracy else 0
            speedup = baseline_inference_time / result['inference_time'] if result['inference_time'] and baseline_inference_time else 0
            
            # Include compression rate if available in results
            compression_rate = result.get('compression_rate') 

            summary_data.append({
                'model': model_name,
                'config_id': i,
                'params': result['params'],
                'flops': result['flops'],
                'accuracy': result['accuracy'],
                'inference_time': result['inference_time'],
                'params_reduction': params_reduction,
                'flops_reduction': flops_reduction,
                'accuracy_change': accuracy_change,
                'speedup': speedup,
                'compression_rate': compression_rate, # Added
                'baseline_params': baseline_params,
                'baseline_flops': baseline_flops,
                'baseline_accuracy': baseline_accuracy,
                'baseline_inference_time': baseline_inference_time
            })
    
    return pd.DataFrame(summary_data)

# --- Find Pareto Optimal ---
def find_pareto_optimal(df, model_name):
    """Find Pareto-optimal configurations for a given model."""
    model_df = df[df['model'] == model_name].copy()
    
    # We want to maximize these (minimize params/flops, maximize accuracy)
    model_df['neg_params'] = -model_df['params']
    model_df['neg_flops'] = -model_df['flops']
    
    pareto_optimal = []
    if model_df.empty:
        return model_df # Return empty df if no data for model
        
    for idx, row in model_df.iterrows():
        is_dominated = False
        for _, other_row in model_df.iterrows():
            # Check if other_row dominates row
            # A configuration dominates if it's better or equal in all objectives 
            # and strictly better in at least one.
            if (other_row['neg_params'] >= row['neg_params'] and 
                other_row['neg_flops'] >= row['neg_flops'] and 
                other_row['accuracy'] >= row['accuracy'] and 
                (other_row['neg_params'] > row['neg_params'] or 
                 other_row['neg_flops'] > row['neg_flops'] or 
                 other_row['accuracy'] > row['accuracy'])): 
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimal.append(idx)
    
    # Check if pareto_optimal list is empty before indexing
    if not pareto_optimal:
        return pd.DataFrame(columns=model_df.columns) # Return empty df with same columns
        
    return model_df.loc[pareto_optimal].sort_values('params_reduction', ascending=False)

# --- Plotting Functions --- 
def plot_pareto_frontier(df, output_dir):
    """Plot Pareto frontier for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    model_names = ['resnet', 'alexnet', 'vgg']
    
    # Find global min/max for consistent axes
    all_acc = df['accuracy'].dropna()
    min_acc = all_acc.min() if not all_acc.empty else 0.9
    max_acc = all_acc.max() if not all_acc.empty else 1.0
    all_pr = df['params_reduction'].dropna()
    min_pr = all_pr.min() if not all_pr.empty else 0
    max_pr = all_pr.max() if not all_pr.empty else 1
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_df = df[df['model'] == model_name].dropna(subset=['params_reduction', 'accuracy'])
        if model_df.empty:
            ax.set_title(f"{model_name.capitalize()} (No Data)")
            continue
            
        pareto_df = find_pareto_optimal(df, model_name).dropna(subset=['params_reduction', 'accuracy'])
        
        scatter = ax.scatter(
            model_df['params_reduction'], 
            model_df['accuracy'], 
            alpha=0.5, 
            label='All Configs'
        )
        
        if not pareto_df.empty:
            ax.scatter(
                pareto_df['params_reduction'], 
                pareto_df['accuracy'], 
                color='red', 
                s=100, 
                label='Pareto Optimal',
                zorder=5 # Ensure Pareto points are on top
            )
            
            # Connect Pareto points with a line
            pareto_df_sorted = pareto_df.sort_values('params_reduction')
            ax.plot(
                pareto_df_sorted['params_reduction'], 
                pareto_df_sorted['accuracy'], 
                'r--',
                zorder=4
            )
        
        ax.set_title(f"{model_name.capitalize()} Pareto Frontier")
        ax.set_xlabel("Parameters Reduction")
        if i == 0:
             ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
        # Set consistent limits
        ax.set_xlim(min_pr - 0.05, max_pr + 0.05)
        ax.set_ylim(min_acc - 0.01, max_acc + 0.01) 
    
    plt.suptitle("Accuracy vs. Parameter Reduction Pareto Frontiers", fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout
    # Use FIGS_DIR for saving
    plt.savefig(os.path.join(output_dir, 'pareto_frontiers.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(df, output_dir):
    """Create radar charts comparing the best configurations for each model."""
    # Radar chart setup uses the global radar_factory function
    best_configs = []
    for model_name in ['resnet', 'alexnet', 'vgg']:
        model_df = df[df['model'] == model_name]
        if model_df.empty or 'baseline_accuracy' not in model_df.columns or model_df['baseline_accuracy'].isnull().all():
            print(f"Skipping {model_name} for radar chart due to missing data or baseline.")
            continue
            
        # Configs with accuracy within 2% of baseline
        viable_configs = model_df[model_df['accuracy'] >= model_df['baseline_accuracy'] - 0.02]
        
        if not viable_configs.empty:
            # Get config with highest params reduction among viable ones
            best_config = viable_configs.loc[viable_configs['params_reduction'].idxmax()]
            best_configs.append(best_config)
        else:
            print(f"No viable configurations (Accuracy >= Baseline - 0.02) found for {model_name}")
            # Optionally, add the best overall params reduction config if no viable ones found
            if not model_df.empty:
                 best_overall = model_df.loc[model_df['params_reduction'].idxmax()]
                 print(f"  -> Adding best overall params reduction config for {model_name} instead.")
                 best_configs.append(best_overall)

    if not best_configs:
        print("No configurations found for radar chart after checks.")
        return
    
    # Prepare data for radar chart
    metrics = ['Params Red.', 'FLOPs Red.', 'Acc. Ret.', 'Speedup']
    N = len(metrics)
    theta = radar_factory(N)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    
    data_for_radar = []
    labels = []
    for config in best_configs:
         # Ensure baseline accuracy exists for retention calculation
        baseline_acc = config.get('baseline_accuracy')
        if baseline_acc is None or baseline_acc == 0:
            print(f"Warning: Baseline accuracy missing or zero for {config['model']}, skipping accuracy retention.")
            accuracy_retention = 0 # or handle differently
        else:
            accuracy_retention = config['accuracy'] / baseline_acc

        params_reduction = config.get('params_reduction', 0)
        flops_reduction = config.get('flops_reduction', 0)
        # Cap speedup at 2x for better visualization, handle potential division by zero or missing value
        speedup_raw = config.get('speedup')
        speedup_norm = min(speedup_raw, 2) / 2 if speedup_raw else 0 
        
        values = [params_reduction, flops_reduction, accuracy_retention, speedup_norm]
        data_for_radar.append(values)
        labels.append(config['model'].capitalize())

    # Plotting
    colors = plt.cm.get_cmap('Set2', len(labels))
    for i, (d, label) in enumerate(zip(data_for_radar, labels)):
        ax.plot(theta, d, color=colors(i), marker='o', label=label)
        ax.fill(theta, d, facecolor=colors(i), alpha=0.25)

    ax.set_varlabels(metrics)
    ax.set_yticks(np.arange(0, 1.1, 0.2)) # Set radial ticks from 0 to 1
    ax.set_ylim(0, 1) # Ensure the axis limits are set
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title('Best Model Configurations Comparison\n(Max Params Reduction with Acc >= Baseline-2%)', position=(0.5, 1.1))
    
    # Use FIGS_DIR for saving
    plt.savefig(os.path.join(output_dir, 'model_comparison_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_tradeoff_plot(df, output_dir):
    """Create a plot showing the tradeoff between params reduction, FLOPs reduction, and accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    model_names = ['resnet', 'alexnet', 'vgg']
    
    # Find global min/max for consistent axes
    all_pr = df['params_reduction'].dropna()
    all_fr = df['flops_reduction'].dropna()
    all_acc = df['accuracy'].dropna()
    min_pr = all_pr.min() if not all_pr.empty else 0
    max_pr = all_pr.max() if not all_pr.empty else 1
    min_fr = all_fr.min() if not all_fr.empty else 0
    max_fr = all_fr.max() if not all_fr.empty else 1
    min_acc_c = all_acc.min() if not all_acc.empty else 0.9
    max_acc_c = all_acc.max() if not all_acc.empty else 1.0

    norm = plt.Normalize(vmin=min_acc_c, vmax=max_acc_c)
    cmap = plt.cm.viridis
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_df = df[df['model'] == model_name].dropna(subset=['params_reduction', 'flops_reduction', 'accuracy'])
        
        if model_df.empty:
            ax.set_title(f"{model_name.capitalize()} (No Data)")
            continue

        scatter = ax.scatter(
            model_df['params_reduction'], 
            model_df['flops_reduction'], 
            c=model_df['accuracy'], 
            cmap=cmap, 
            norm=norm,
            s=100,
            alpha=0.7
        )
        
        ax.set_title(f"{model_name.capitalize()} Tradeoff")
        ax.set_xlabel("Parameters Reduction")
        if i == 0:
            ax.set_ylabel("FLOPs Reduction")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(min_pr - 0.05, max_pr + 0.05)
        ax.set_ylim(min_fr - 0.05, max_fr + 0.05)
        
    # Add a single colorbar for the entire figure
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', label='Accuracy', fraction=0.02, pad=0.04)
    
    plt.suptitle("Accuracy vs. Parameter and FLOPs Reduction", fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    # Use FIGS_DIR for saving
    plt.savefig(os.path.join(output_dir, 'tradeoff_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(df, output_dir):
    """Create and save a summary table of the best configurations for each model."""
    best_configs = []
    
    # Define different criteria for selecting the best config
    criteria = [
        {"name": "Best Params Red.", "sort_by": "params_reduction", "ascending": False, 
         "condition": lambda x: x['accuracy'] >= x.get('baseline_accuracy', 0) - 0.02},
        {"name": "Best FLOPs Red.", "sort_by": "flops_reduction", "ascending": False, 
         "condition": lambda x: x['accuracy'] >= x.get('baseline_accuracy', 0) - 0.02},
        {"name": "Best Speedup", "sort_by": "speedup", "ascending": False, 
         "condition": lambda x: x['accuracy'] >= x.get('baseline_accuracy', 0) - 0.02},
        {"name": "Highest Accuracy", "sort_by": "accuracy", "ascending": False, 
         "condition": lambda x: x['params_reduction'] >= 0.3}  # At least 30% params reduction
    ]
    
    # For each model, find the best config according to each criterion
    for model_name in ['resnet', 'alexnet', 'vgg']:
        model_df = df[df['model'] == model_name].dropna(subset=['accuracy', 'params_reduction', 'flops_reduction', 'speedup'])
        if model_df.empty or 'baseline_accuracy' not in model_df.columns:
             print(f"Skipping {model_name} for summary table due to missing data or baseline.")
             continue
             
        for criterion in criteria:
            # Apply condition filter
            # Need to handle potential missing baseline_accuracy during condition check
            filtered_df = model_df[model_df.apply(lambda x: criterion["condition"](x) if pd.notna(x.get('baseline_accuracy')) else False, axis=1)]
            
            if not filtered_df.empty:
                # Sort and get the best
                sorted_df = filtered_df.sort_values(criterion["sort_by"], ascending=criterion["ascending"])
                best_config = sorted_df.iloc[0].copy()
                best_config["criterion"] = criterion["name"]
                best_configs.append(best_config)
            else:
                 print(f"No config found for {model_name} matching criterion: {criterion['name']}")

    if not best_configs:
        print("No best configurations found to create summary table.")
        return pd.DataFrame()

    # Create a DataFrame from the best configs
    summary_df = pd.DataFrame(best_configs)
    
    # Select and rename columns for the table
    table_df = summary_df[[
        'model', 'criterion', 'params_reduction', 'flops_reduction', 
        'accuracy', 'speedup', 'params', 'flops', 'compression_rate' # Added compression rate
    ]].copy()
    
    table_df.rename(columns={
        'model': 'Model',
        'criterion': 'Criterion',
        'params_reduction': 'Params Red.',
        'flops_reduction': 'FLOPs Red.',
        'accuracy': 'Accuracy',
        'speedup': 'Speedup',
        'params': 'Parameters',
        'flops': 'FLOPs',
        'compression_rate': 'Comp. Rate' # Added
    }, inplace=True)
    
    # Format numbers
    format_dict = {
        'Params Red.': '{:.2%}',
        'FLOPs Red.': '{:.2%}',
        'Accuracy': '{:.3f}',
        'Speedup': '{:.2f}x',
        'Parameters': '{:,.0f}',
        'FLOPs': '{:,.0f}',
        'Comp. Rate': '{:.2f}' # Added
    }
    
    # Apply formatting carefully, handling potential NaNs
    for col, fmt in format_dict.items():
        if col in table_df.columns:
             table_df[col] = table_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
        else:
            print(f"Warning: Column '{col}' not found for formatting in summary table.")

    # Ensure model names are capitalized
    table_df['Model'] = table_df['Model'].str.capitalize()

    # Save as HTML
    html_table = table_df.to_html(index=False, border=1, classes=['dataframe', 'styled-table'])
    # Add some basic CSS for better appearance
    html_content = f"""
    <html>
    <head>
    <style>
      .styled-table {{
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
      }}
      .styled-table thead tr {{
        background-color: #009879;
        color: #ffffff;
        text-align: left;
      }}
      .styled-table th,
      .styled-table td {{
        padding: 12px 15px;
        border: 1px solid #dddddd;
      }}
      .styled-table tbody tr {{
        border-bottom: 1px solid #dddddd;
      }}
      .styled-table tbody tr:nth-of-type(even) {{
        background-color: #f3f3f3;
      }}
      .styled-table tbody tr:last-of-type {{
        border-bottom: 2px solid #009879;
      }}
    </style>
    </head>
    <body>
    {html_table}
    </body>
    </html>
    """
    # Use TABLES_DIR for saving
    with open(os.path.join(output_dir, 'best_configs_summary.html'), 'w') as f:
        f.write(html_content)
    
    # Save as CSV
    # Use TABLES_DIR for saving
    table_df.to_csv(os.path.join(output_dir, 'best_configs_summary.csv'), index=False)
    
    # Create a text version with tabulate
    text_table = tabulate(table_df, headers='keys', tablefmt='pretty', showindex=False)
    # Use TABLES_DIR for saving
    with open(os.path.join(output_dir, 'best_configs_summary.txt'), 'w') as f:
        f.write(text_table)
    
    return table_df

def plot_accuracy_vs_computation(df, output_dir):
    """Plot accuracy vs computation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Find global min/max for consistent axes
    all_acc = df['accuracy'].dropna() * 100
    all_params = df['params'].dropna() / 1e6
    all_flops = df['flops'].dropna() / 1e9
    min_acc = all_acc.min() if not all_acc.empty else 90
    max_acc = all_acc.max() if not all_acc.empty else 100
    min_params = all_params.min() if not all_params.empty else 0
    max_params = all_params.max() if not all_params.empty else 50
    min_flops = all_flops.min() if not all_flops.empty else 0
    max_flops = all_flops.max() if not all_flops.empty else 10
    
    # Params vs Accuracy
    has_baseline_label_p = False
    for model_name in ['resnet', 'alexnet', 'vgg']:
        model_df = df[df['model'] == model_name].dropna(subset=['params', 'accuracy', 'baseline_params', 'baseline_accuracy'])
        if model_df.empty:
            continue
            
        axes[0].scatter(
            model_df['params'] / 1e6,  # Convert to millions 
            model_df['accuracy'] * 100,  # Convert to percentage
            label=model_name.capitalize(),
            alpha=0.7,
            s=80
        )
        
        # Mark baseline (only the first row for each model)
        baseline = model_df.iloc[0]
        baseline_label = f"Baseline" if not has_baseline_label_p else ""
        has_baseline_label_p = True
        axes[0].scatter(
            baseline['baseline_params'] / 1e6,
            baseline['baseline_accuracy'] * 100,
            marker='*',
            color=plt.cm.get_cmap('Set2')(list(df['model'].unique()).index(model_name)), # Match color
            s=250,
            edgecolor='black',
            linewidth=1.5,
            label=baseline_label,
            zorder=5
        )
    
    axes[0].set_title('Accuracy vs Parameters')
    axes[0].set_xlabel('Parameters (Millions)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    axes[0].set_xlim(min_params * 0.9, max_params * 1.1)
    axes[0].set_ylim(min_acc * 0.99, max_acc * 1.01)

    # FLOPs vs Accuracy
    has_baseline_label_f = False
    for model_name in ['resnet', 'alexnet', 'vgg']:
        model_df = df[df['model'] == model_name].dropna(subset=['flops', 'accuracy', 'baseline_flops', 'baseline_accuracy'])
        if model_df.empty:
            continue
            
        axes[1].scatter(
            model_df['flops'] / 1e9,  # Convert to billions
            model_df['accuracy'] * 100,  # Convert to percentage
            label=model_name.capitalize(),
            alpha=0.7,
            s=80
        )
        
        # Mark baseline
        baseline = model_df.iloc[0]
        baseline_label = f"Baseline" if not has_baseline_label_f else ""
        has_baseline_label_f = True
        axes[1].scatter(
            baseline['baseline_flops'] / 1e9,
            baseline['baseline_accuracy'] * 100,
            marker='*',
            color=plt.cm.get_cmap('Set2')(list(df['model'].unique()).index(model_name)), # Match color
            s=250,
            edgecolor='black',
            linewidth=1.5,
            label=baseline_label,
            zorder=5
        )
    
    axes[1].set_title('Accuracy vs FLOPs')
    axes[1].set_xlabel('FLOPs (Billions)')
    # axes[1].set_ylabel('Accuracy (%)') # Removed as shared with left plot
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    axes[1].set_xlim(min_flops * 0.9, max_flops * 1.1)
    axes[1].set_ylim(min_acc * 0.99, max_acc * 1.01)
    
    plt.suptitle("Accuracy vs. Computational Cost", fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    # Use FIGS_DIR for saving
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_computation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboard(df, output_dir):
    """Create a comprehensive dashboard of all the visualizations."""
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 18)) # Increased height for table
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.8]) # Adjusted grid and height ratios
    
    # --- Row 1: Pareto Frontiers --- 
    model_names = ['resnet', 'alexnet', 'vgg']
    all_acc = df['accuracy'].dropna()
    min_acc_p = all_acc.min() if not all_acc.empty else 0.9
    max_acc_p = all_acc.max() if not all_acc.empty else 1.0
    all_pr = df['params_reduction'].dropna()
    min_pr = all_pr.min() if not all_pr.empty else 0
    max_pr = all_pr.max() if not all_pr.empty else 1

    for i, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, i])
        model_df = df[df['model'] == model_name].dropna(subset=['params_reduction', 'accuracy'])
        if model_df.empty:
            ax.set_title(f"{model_name.capitalize()} Pareto (No Data)")
            continue
            
        pareto_df = find_pareto_optimal(df, model_name).dropna(subset=['params_reduction', 'accuracy'])
        
        ax.scatter(
            model_df['params_reduction'], 
            model_df['accuracy'], 
            alpha=0.5, 
            label='All Configs'
        )
        
        if not pareto_df.empty:
            ax.scatter(
                pareto_df['params_reduction'], 
                pareto_df['accuracy'], 
                color='red', 
                s=100, 
                label='Pareto Optimal',
                zorder=5
            )
            pareto_df_sorted = pareto_df.sort_values('params_reduction')
            ax.plot(
                pareto_df_sorted['params_reduction'], 
                pareto_df_sorted['accuracy'], 
                'r--',
                zorder=4
            )
        
        ax.set_title(f"{model_name.capitalize()} Pareto Frontier")
        ax.set_xlabel("Params Reduction")
        if i == 0: ax.set_ylabel("Accuracy")
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True)
        ax.set_xlim(min_pr - 0.05, max_pr + 0.05)
        ax.set_ylim(min_acc_p - 0.01, max_acc_p + 0.01)

    # --- Row 2: Accuracy vs Computation --- 
    all_acc = df['accuracy'].dropna() * 100
    all_params = df['params'].dropna() / 1e6
    all_flops = df['flops'].dropna() / 1e9
    min_acc = all_acc.min() if not all_acc.empty else 90
    max_acc = all_acc.max() if not all_acc.empty else 100
    min_params = all_params.min() if not all_params.empty else 0
    max_params = all_params.max() if not all_params.empty else 50
    min_flops = all_flops.min() if not all_flops.empty else 0
    max_flops = all_flops.max() if not all_flops.empty else 10

    # Accuracy vs Parameters
    ax = fig.add_subplot(gs[1, 0])
    has_baseline_label_p = False
    for model_name in model_names:
        model_df = df[df['model'] == model_name].dropna(subset=['params', 'accuracy', 'baseline_params', 'baseline_accuracy'])
        if model_df.empty: continue
        ax.scatter(model_df['params'] / 1e6, model_df['accuracy'] * 100, label=model_name.capitalize(), alpha=0.7)
        baseline = model_df.iloc[0]
        baseline_label = f"Baseline" if not has_baseline_label_p else ""
        has_baseline_label_p = True
        ax.scatter(baseline['baseline_params'] / 1e6, baseline['baseline_accuracy'] * 100, marker='*',
                   color=plt.cm.get_cmap('Set2')(list(df['model'].unique()).index(model_name)), s=200,
                   edgecolor='black', linewidth=1.5, label=baseline_label, zorder=5)
    ax.set_title('Accuracy vs Parameters')
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlim(min_params * 0.9, max_params * 1.1)
    ax.set_ylim(min_acc * 0.99, max_acc * 1.01)

    # Accuracy vs FLOPs
    ax = fig.add_subplot(gs[1, 1])
    has_baseline_label_f = False
    for model_name in model_names:
        model_df = df[df['model'] == model_name].dropna(subset=['flops', 'accuracy', 'baseline_flops', 'baseline_accuracy'])
        if model_df.empty: continue
        ax.scatter(model_df['flops'] / 1e9, model_df['accuracy'] * 100, label=model_name.capitalize(), alpha=0.7)
        baseline = model_df.iloc[0]
        baseline_label = f"Baseline" if not has_baseline_label_f else ""
        has_baseline_label_f = True
        ax.scatter(baseline['baseline_flops'] / 1e9, baseline['baseline_accuracy'] * 100, marker='*',
                   color=plt.cm.get_cmap('Set2')(list(df['model'].unique()).index(model_name)), s=200,
                   edgecolor='black', linewidth=1.5, label=baseline_label, zorder=5)
    ax.set_title('Accuracy vs FLOPs')
    ax.set_xlabel('FLOPs (Billions)')
    # ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlim(min_flops * 0.9, max_flops * 1.1)
    ax.set_ylim(min_acc * 0.99, max_acc * 1.01)

    # Radar Chart
    ax = fig.add_subplot(gs[1, 2], projection='radar')
    best_configs = []
    for model_name in model_names:
        model_df = df[df['model'] == model_name]
        if model_df.empty or 'baseline_accuracy' not in model_df.columns or model_df['baseline_accuracy'].isnull().all(): continue
        viable_configs = model_df[model_df['accuracy'] >= model_df['baseline_accuracy'] - 0.02]
        if not viable_configs.empty:
            best_config = viable_configs.loc[viable_configs['params_reduction'].idxmax()]
            best_configs.append(best_config)
        elif not model_df.empty:
             best_overall = model_df.loc[model_df['params_reduction'].idxmax()]
             best_configs.append(best_overall)
             
    if best_configs:
        metrics = ['Params Red.', 'FLOPs Red.', 'Acc. Ret.', 'Speedup']
        N = len(metrics)
        theta = radar_factory(N) # Use global radar_factory
        data_for_radar = []
        labels = []
        for config in best_configs:
            baseline_acc = config.get('baseline_accuracy')
            accuracy_retention = (config['accuracy'] / baseline_acc) if baseline_acc else 0
            params_reduction = config.get('params_reduction', 0)
            flops_reduction = config.get('flops_reduction', 0)
            speedup_raw = config.get('speedup')
            speedup_norm = min(speedup_raw, 2) / 2 if speedup_raw else 0 
            values = [params_reduction, flops_reduction, accuracy_retention, speedup_norm]
            data_for_radar.append(values)
            labels.append(config['model'].capitalize())
        colors = plt.cm.get_cmap('Set2', len(labels))
        for i, (d, label) in enumerate(zip(data_for_radar, labels)):
            ax.plot(theta, d, color=colors(i), marker='o', label=label)
            ax.fill(theta, d, facecolor=colors(i), alpha=0.25)
        ax.set_varlabels(metrics)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))
        ax.set_title('Best Config Comparison', position=(0.5, 1.15))
    else:
         ax.set_title('Best Config Comparison (No Data)')
         ax.axis('off')

    # --- Row 3: Summary Table --- 
    ax = fig.add_subplot(gs[2, :]) # Span table across the bottom row
    summary_table_df = create_summary_table(df, TABLES_DIR) # Get the formatted df
    if not summary_table_df.empty:
        ax.axis('off')
        table = ax.table(
            cellText=summary_table_df.values,
            colLabels=summary_table_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('Best Configurations Summary', pad=20)
    else:
        ax.text(0.5, 0.5, 'Summary Table Not Available (Missing Data)', ha='center', va='center')
        ax.axis('off')
        ax.set_title('Best Configurations Summary')

    fig.suptitle('HTD Experiment Comprehensive Analysis Dashboard', fontsize=24, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    # Use FIGS_DIR for saving
    plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load results and corresponding baselines
    results, baselines = load_results_and_baseline()
    
    if not results:
        print("No results found. Please run the experiments and jsonify_logs.py first.")
        return
        
    if not baselines or all(v is None for v in baselines.values()):
        print("Warning: Could not load baseline metrics for any model. Relative analysis will be incomplete.")

    # Create summary dataframe
    df = create_summary_df(results, baselines)
    
    if df.empty:
        print("Summary dataframe is empty, likely due to missing baseline data. Aborting analysis.")
        return

    # Save raw combined data (including baselines)
    df.to_csv(os.path.join(TABLES_DIR, 'all_results_summary.csv'), index=False)
    
    # Create visualizations (saved to FIGS_DIR)
    print("Creating visualizations...")
    plot_pareto_frontier(df, FIGS_DIR)
    plot_radar_chart(df, FIGS_DIR)
    create_tradeoff_plot(df, FIGS_DIR)
    plot_accuracy_vs_computation(df, FIGS_DIR)
    create_comprehensive_dashboard(df, FIGS_DIR) # Pass FIGS_DIR
    
    # Create summary table (saved to TABLES_DIR)
    print("Creating summary table...")
    summary_table = create_summary_table(df, TABLES_DIR) # Pass TABLES_DIR
    print("--- Best Configurations Summary ---")
    print(tabulate(summary_table, headers='keys', tablefmt='pretty', showindex=False))
    
    print(f"\nAnalysis complete. Figures saved to {FIGS_DIR}, Tables saved to {TABLES_DIR}")
    
if __name__ == "__main__":
    main()