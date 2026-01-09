"""
Streamlit Comparator App

A tool for comparing evaluation runs across different experiments.
Allows side-by-side comparison of generated corrections, auxiliary metrics,
and evaluation results for student submissions.
"""

import streamlit as st
from utils import (
    load_config, 
    get_submissions, 
    get_submission_data, 
    get_available_runs, 
    get_run_data,
    get_prompts_data,
    format_timestamp
)


# Page configuration
st.set_page_config(
    page_title="Evaluation Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .run-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .metric-box {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .score-display {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .submission-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #0f3460;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .submission-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.2);
    }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* Ensure text wraps in expanders */
    .stExpander div[data-testid="stMarkdownContainer"] {
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .stExpander p, .stExpander code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'selected_submission' not in st.session_state:
        st.session_state.selected_submission = None
    if 'selected_runs' not in st.session_state:
        st.session_state.selected_runs = []
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    # Sync mode state
    if 'sync_enabled' not in st.session_state:
        st.session_state.sync_enabled = False
    if 'sync_active_tab' not in st.session_state:
        st.session_state.sync_active_tab = "📝 Corrections"  # Default tab
    if 'sync_corrections_expanded' not in st.session_state:
        st.session_state.sync_corrections_expanded = False
    if 'sync_prompts_expanded' not in st.session_state:
        st.session_state.sync_prompts_expanded = False
    if 'sync_metrics_expanded' not in st.session_state:
        st.session_state.sync_metrics_expanded = {}  # metric_name -> expanded state
    if 'sync_eval_explanations_expanded' not in st.session_state:
        st.session_state.sync_eval_explanations_expanded = {}  # explanation_name -> expanded state
    # Per-item sync state (individual corrections/prompts by requirement key)
    if 'sync_correction_items' not in st.session_state:
        st.session_state.sync_correction_items = {}  # req_key -> expanded state
    if 'sync_prompt_items' not in st.session_state:
        st.session_state.sync_prompt_items = {}  # req_key -> expanded state
    if 'sync_render_version' not in st.session_state:
        st.session_state.sync_render_version = 0  # Increment to force expander re-render
    # Per-run expander state (for Open All/Close All in non-sync mode)
    if 'corr_expander_states' not in st.session_state:
        st.session_state.corr_expander_states = {}  # run_key -> {req_key -> expanded}
    if 'prompt_expander_states' not in st.session_state:
        st.session_state.prompt_expander_states = {}  # run_key -> {req_key -> expanded}


def navigate_to_submission(submission_name: str):
    """Navigate to submission page."""
    st.session_state.current_page = 'submission'
    st.session_state.selected_submission = submission_name
    st.session_state.selected_runs = []


def navigate_to_home():
    """Navigate back to home page."""
    st.session_state.current_page = 'home'
    st.session_state.selected_submission = None
    st.session_state.selected_runs = []


def add_run(experiment_type: str, timestamp: str):
    """Add a run to the comparison."""
    run_key = f"{experiment_type}/{timestamp}"
    if run_key not in st.session_state.selected_runs:
        st.session_state.selected_runs.append(run_key)


def remove_run(run_key: str):
    """Remove a run from the comparison."""
    if run_key in st.session_state.selected_runs:
        st.session_state.selected_runs.remove(run_key)


def toggle_sync_mode(enabled: bool):
    """Toggle sync mode and normalize state when enabling."""
    st.session_state.sync_enabled = enabled
    if enabled:
        # Normalize: close all expanders and select corrections tab
        st.session_state.sync_active_tab = "📝 Corrections"
        st.session_state.sync_corrections_expanded = False
        st.session_state.sync_prompts_expanded = False
        st.session_state.sync_metrics_expanded = {}
        st.session_state.sync_eval_explanations_expanded = {}
        st.session_state.sync_correction_items = {}
        st.session_state.sync_prompt_items = {}
        
        # Also reset per-run state keys to match sync state
        for run_key in st.session_state.selected_runs:
            corr_key = f"corr_expanded_{run_key}"
            prompt_key = f"prompt_expanded_{run_key}"
            if corr_key in st.session_state:
                st.session_state[corr_key] = False
            if prompt_key in st.session_state:
                st.session_state[prompt_key] = False


def sync_action(action_type: str, key: str = None, value: bool = None):
    """Handle synchronized actions across all runs."""
    if not st.session_state.sync_enabled:
        return
    
    if action_type == "tab":
        st.session_state.sync_active_tab = key
    elif action_type == "corrections_expand":
        st.session_state.sync_corrections_expanded = value
    elif action_type == "prompts_expand":
        st.session_state.sync_prompts_expanded = value
    elif action_type == "metric_expand":
        st.session_state.sync_metrics_expanded[key] = value
    elif action_type == "explanation_expand":
        st.session_state.sync_eval_explanations_expanded[key] = value


def render_home_page():
    """Render the home page with submission list."""
    st.markdown("# 📊 Evaluation Comparator")
    st.markdown("### Select a submission to compare evaluation runs")
    
    config = st.session_state.config
    submissions = get_submissions(config)
    
    if not submissions:
        st.warning("No submissions found. Check the assignment_path in streamlit.yaml")
        return
    
    st.markdown(f"**Assignment folder:** `{config['assignment_path']}`")
    st.markdown(f"**Found {len(submissions)} submissions**")
    st.divider()
    
    # Display submissions in a grid
    cols = st.columns(4)
    for i, submission in enumerate(submissions):
        with cols[i % 4]:
            if st.button(
                f"📝 {submission}",
                key=f"sub_{submission}",
                use_container_width=True
            ):
                navigate_to_submission(submission)
                st.rerun()


def render_reference_correction(reference):
    """Render reference correction in a readable format."""
    if reference is None:
        st.info("No reference correction available")
        return
    
    if isinstance(reference, list):
        for i, item in enumerate(reference, 1):
            st.markdown(f"**{i}.** {item}")
    elif isinstance(reference, dict) and 'corrections' in reference:
        for i, item in enumerate(reference['corrections'], 1):
            st.markdown(f"**{i}.** {item}")
    else:
        st.text(str(reference))


def render_run_data(run_data: dict, run_key: str):
    """Render the data for a single run."""
    if run_data is None:
        st.error("Could not load run data")
        return
    
    sync_mode = st.session_state.sync_enabled
    
    # Header with remove button
    col1, col2 = st.columns([5, 1])
    with col1:
        exp_type, timestamp = run_key.split('/')
        st.markdown(f"### 🔬 {exp_type}")
        st.caption(f"📅 {format_timestamp(timestamp)}")
    with col2:
        if st.button("❌", key=f"remove_{run_key}"):
            remove_run(run_key)
            st.rerun()
    
    # Tab options
    tab_options = ["📝 Corrections", "📊 Metrics", "🎯 Evaluation", "🧠 Prompts"]
    
    # When sync mode is ON, use a shared state for tab selection
    # When sync mode is OFF, use st.tabs() as normal (per-run tab state)
    if sync_mode:
        # In sync mode, display the content based on shared sync_active_tab state
        # The tab selector is rendered once at the top of the Compare section
        active_tab = st.session_state.sync_active_tab
    else:
        # In non-sync mode, use regular tabs
        tab1, tab2, tab3, tab4 = st.tabs(tab_options)
        # The content is rendered inside each tab context
        with tab1:
            _render_corrections_content(run_data, run_key, sync_mode)
        with tab2:
            _render_metrics_content(run_data, run_key, sync_mode)
        with tab3:
            _render_evaluation_content(run_data, run_key, sync_mode)
        with tab4:
            _render_prompts_content(run_data, run_key, sync_mode)
        return  # Exit early for non-sync mode
    
    # Sync mode: render content based on shared active_tab
    if active_tab == "📝 Corrections":
        _render_corrections_content(run_data, run_key, sync_mode)
    elif active_tab == "📊 Metrics":
        _render_metrics_content(run_data, run_key, sync_mode)
    elif active_tab == "🎯 Evaluation":
        _render_evaluation_content(run_data, run_key, sync_mode)
    elif active_tab == "🧠 Prompts":
        _render_prompts_content(run_data, run_key, sync_mode)


def _render_corrections_content(run_data: dict, run_key: str, sync_mode: bool):
    """Render corrections tab content."""
    corrections = run_data.get('generated_corrections')
    if corrections:
        corr_list = corrections.get('corrections', [])
        if corr_list:
            # Initialize per-run state if needed
            if run_key not in st.session_state.corr_expander_states:
                st.session_state.corr_expander_states[run_key] = {}
            
            # Build list of req_keys for this run
            req_keys = []
            for corr in corr_list:
                req = corr.get('requirement', {})
                req_key = f"{req.get('function', 'unknown')}_{req.get('type', 'unknown')}_{req.get('requirement', '')[:50]}"
                req_keys.append(req_key)
                # Initialize to True (expanded) by default if not set
                if req_key not in st.session_state.corr_expander_states[run_key]:
                    st.session_state.corr_expander_states[run_key][req_key] = True
            
            # Toggle buttons for Open All / Close All
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("📂 Open All", key=f"open_corr_{run_key}"):
                    if sync_mode:
                        # Set all correction items to expanded (shared across runs)
                        for req_key in req_keys:
                            st.session_state.sync_correction_items[req_key] = True
                    else:
                        # Set all correction items to expanded (per-run)
                        for req_key in req_keys:
                            st.session_state.corr_expander_states[run_key][req_key] = True
                    st.rerun()
            with col2:
                if st.button("📁 Close All", key=f"close_corr_{run_key}"):
                    if sync_mode:
                        # Set all correction items to collapsed (shared across runs)
                        for req_key in req_keys:
                            st.session_state.sync_correction_items[req_key] = False
                    else:
                        # Set all correction items to collapsed (per-run)
                        for req_key in req_keys:
                            st.session_state.corr_expander_states[run_key][req_key] = False
                    st.rerun()
            
            if sync_mode:
                st.caption("💡 Use the toggle button inside each expander for reliable sync across runs.")
            
            for i, corr in enumerate(corr_list):
                req = corr.get('requirement', {})
                func_name = req.get('function', 'Unknown')
                req_type = req.get('type', '')
                req_text = req.get('requirement', 'N/A')
                req_key = f"{func_name}_{req_type}_{req_text[:50]}"
                
                # Truncate long requirements for the title
                req_preview = req_text[:80] + "..." if len(req_text) > 80 else req_text
                title = f"**{func_name}** ({req_type}) - {req_preview}"
                
                # Determine expanded state
                if sync_mode:
                    # Use shared per-item state
                    item_expanded = st.session_state.sync_correction_items.get(req_key, False)
                else:
                    # Use per-run state
                    item_expanded = st.session_state.corr_expander_states[run_key].get(req_key, True)
                
                with st.expander(title, expanded=item_expanded):
                    # Toggle button to track state changes
                    toggle_label = "🔒 Collapse" if item_expanded else "🔓 Expand"
                    if st.button(toggle_label, key=f"toggle_corr_{run_key}_{i}"):
                        if sync_mode:
                            st.session_state.sync_correction_items[req_key] = not item_expanded
                        else:
                            st.session_state.corr_expander_states[run_key][req_key] = not item_expanded
                        st.rerun()
                    
                    st.markdown("**Result:**")
                    st.markdown(corr.get('result', 'No result'))
        else:
            st.info("No corrections in this run")
    else:
        st.info("No corrections data available")


def _render_metrics_content(run_data: dict, run_key: str, sync_mode: bool):
    """Render metrics tab content."""
    aux_metrics = run_data.get('auxiliary_metrics')
    if aux_metrics:
        metrics = aux_metrics.get('auxiliary_metrics', aux_metrics)
        if isinstance(metrics, dict):
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                if metric_name != 'timings':
                    # Determine expanded state based on sync mode
                    if sync_mode:
                        metric_expanded = st.session_state.sync_metrics_expanded.get(metric_name, False)
                    else:
                        metric_expanded = True
                    
                    with st.expander(f"📈 {metric_name.upper()}", expanded=metric_expanded):
                        # In sync mode, add a toggle button to sync this specific metric
                        if sync_mode:
                            toggle_label = "🔒 Collapse" if metric_expanded else "🔓 Expand"
                            if st.button(toggle_label, key=f"toggle_metric_{run_key}_{i}"):
                                st.session_state.sync_metrics_expanded[metric_name] = not metric_expanded
                                st.session_state.sync_render_version += 1
                                st.rerun()
                        
                        if isinstance(metric_value, str):
                            st.markdown(metric_value)
                        elif isinstance(metric_value, list):
                            for item in metric_value:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(f"```json\n{metric_value}\n```")
        else:
            st.json(metrics)
    else:
        st.info("No auxiliary metrics available")


def _render_evaluation_content(run_data: dict, run_key: str, sync_mode: bool):
    """Render evaluation tab content."""
    eval_results = run_data.get('evaluation_results')
    if eval_results:
        # Overall score
        overall = eval_results.get('overall_score')
        if overall is not None:
            st.metric("Overall Score", f"{overall:.3f}")
        
        # Individual scores
        scores = eval_results.get('scores', {})
        if scores:
            st.markdown("**Individual Scores:**")
            score_cols = st.columns(len(scores))
            for i, (name, score) in enumerate(scores.items()):
                with score_cols[i]:
                    st.metric(name, f"{score:.3f}")
        
        # Explanations
        explanations = eval_results.get('explanations', {})
        if explanations:
            st.markdown("**Explanations:**")
            for i, (name, explanation) in enumerate(explanations.items()):
                # Determine expanded state based on sync mode
                if sync_mode:
                    expl_expanded = st.session_state.sync_eval_explanations_expanded.get(name, False)
                else:
                    expl_expanded = False
                
                with st.expander(f"💡 {name}", expanded=expl_expanded):
                    # In sync mode, add a toggle button to sync this specific explanation
                    if sync_mode:
                        toggle_label = "🔒 Collapse" if expl_expanded else "🔓 Expand"
                        if st.button(toggle_label, key=f"toggle_expl_{run_key}_{i}"):
                            st.session_state.sync_eval_explanations_expanded[name] = not expl_expanded
                            st.session_state.sync_render_version += 1
                            st.rerun()
                    
                    st.markdown(explanation)
        
        # Timings
        timings = eval_results.get('timings', {})
        if timings:
            with st.expander("⏱️ Timings"):
                for name, time_val in timings.items():
                    st.markdown(f"**{name}:** {time_val:.2f}s")
    else:
        st.info("No evaluation results available")


def _render_prompts_content(run_data: dict, run_key: str, sync_mode: bool):
    """Render prompts tab content."""
    # Load prompts data (prompts are at timestamp level, not submission level)
    exp_type, timestamp = run_key.split('/')
    prompts = get_prompts_data(st.session_state.config, exp_type, timestamp)
    
    if prompts:
        # Initialize per-run state if needed
        if run_key not in st.session_state.prompt_expander_states:
            st.session_state.prompt_expander_states[run_key] = {}
        
        # Build list of req_keys for this run
        req_keys = []
        for prompt in prompts:
            req = prompt.get('requirement', {})
            req_key = f"{req.get('function', 'unknown')}_{req.get('type', 'unknown')}_{req.get('requirement', '')[:50]}"
            req_keys.append(req_key)
            # Initialize to False (collapsed) by default if not set
            if req_key not in st.session_state.prompt_expander_states[run_key]:
                st.session_state.prompt_expander_states[run_key][req_key] = False
        
        # Toggle buttons for Open All / Close All
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("📂 Open All", key=f"open_prompt_{run_key}"):
                if sync_mode:
                    # Set all prompt items to expanded (shared across runs)
                    for req_key in req_keys:
                        st.session_state.sync_prompt_items[req_key] = True
                else:
                    # Set all prompt items to expanded (per-run)
                    for req_key in req_keys:
                        st.session_state.prompt_expander_states[run_key][req_key] = True
                st.rerun()
        with col2:
            if st.button("📁 Close All", key=f"close_prompt_{run_key}"):
                if sync_mode:
                    # Set all prompt items to collapsed (shared across runs)
                    for req_key in req_keys:
                        st.session_state.sync_prompt_items[req_key] = False
                else:
                    # Set all prompt items to collapsed (per-run)
                    for req_key in req_keys:
                        st.session_state.prompt_expander_states[run_key][req_key] = False
                st.rerun()
        
        if sync_mode:
            st.caption("💡 Use the toggle button inside each expander for reliable sync across runs.")
        
        st.markdown(f"**{len(prompts)} prompts generated**")
        for i, prompt in enumerate(prompts):
            req = prompt.get('requirement', {})
            func_name = req.get('function', 'Unknown')
            req_type = req.get('type', '')
            req_text = req.get('requirement', 'N/A')
            req_key = f"{func_name}_{req_type}_{req_text[:50]}"
            
            # Truncate long requirements for the title
            req_preview = req_text[:80] + "..." if len(req_text) > 80 else req_text
            title = f"**{func_name}** ({req_type}) - {req_preview}"
            
            # Determine expanded state
            if sync_mode:
                item_expanded = st.session_state.sync_prompt_items.get(req_key, False)
            else:
                # Use per-run state
                item_expanded = st.session_state.prompt_expander_states[run_key].get(req_key, False)
            
            with st.expander(title, expanded=item_expanded):
                # Toggle button to track state changes
                toggle_label = "🔒 Collapse" if item_expanded else "🔓 Expand"
                if st.button(toggle_label, key=f"toggle_prompt_{run_key}_{i}"):
                    if sync_mode:
                        st.session_state.sync_prompt_items[req_key] = not item_expanded
                    else:
                        st.session_state.prompt_expander_states[run_key][req_key] = not item_expanded
                    st.rerun()
                
                # Show the jinja template
                st.markdown("**Generated Template:**")
                template = prompt.get('jinja_template', 'No template')
                st.markdown(template)
                
                # Show examples if available
                examples = prompt.get('examples', '')
                if examples:
                    with st.expander("📚 Examples Used", expanded=False):
                        st.markdown(examples)
    else:
        st.info("No prompts data available for this run")


def render_submission_page():
    """Render the submission detail page."""
    submission_name = st.session_state.selected_submission
    config = st.session_state.config
    
    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to_home()
            st.rerun()
    with col2:
        st.markdown(f"# 📄 Submission: {submission_name}")
    
    # Load submission data
    sub_data = get_submission_data(config, submission_name)
    
    # Sidebar for run selection
    with st.sidebar:
        st.markdown("## 🔍 Add Runs to Compare")
        
        available_runs = get_available_runs(config, submission_name)
        
        if not available_runs:
            st.warning("No runs found for this submission")
        else:
            # Experiment type selector
            exp_type = st.selectbox(
                "Experiment Type",
                options=list(available_runs.keys()),
                key="exp_type_select"
            )
            
            if exp_type:
                timestamps = available_runs[exp_type]
                
                # Available runs section (with plus buttons to add)
                st.markdown("#### Available Runs")
                if timestamps:
                    for ts in timestamps:
                        run_key = f"{exp_type}/{ts}"
                        # Only show if not already selected
                        if run_key not in st.session_state.selected_runs:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.caption(format_timestamp(ts))
                            with col2:
                                if st.button("➕", key=f"add_{run_key}", help="Add this run"):
                                    add_run(exp_type, ts)
                                    st.rerun()
                    
                    # Check if all runs are already added
                    all_added = all(f"{exp_type}/{ts}" in st.session_state.selected_runs for ts in timestamps)
                    if all_added:
                        st.success("✓ All runs added")
                else:
                    st.caption("No timestamps available")
        
        st.divider()
        
        # Selected runs section (with cross buttons to remove)
        st.markdown("### Selected Runs")
        if st.session_state.selected_runs:
            for run_key in st.session_state.selected_runs:
                exp, ts = run_key.split('/')
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"{exp[:15]}.. / {format_timestamp(ts)}")
                with col2:
                    if st.button("❌", key=f"remove_{run_key}", help="Remove this run"):
                        remove_run(run_key)
                        st.rerun()
        else:
            st.caption("No runs selected")
    
    # Main content area
    tab_code, tab_ref, tab_compare = st.tabs([
        "📝 Submission Code", 
        "✅ Reference Correction",
        "🔄 Compare Runs"
    ])
    
    with tab_code:
        st.markdown("### Student Code")
        if sub_data['code']:
            st.code(sub_data['code'], language='python', line_numbers=True)
        else:
            st.warning("Code file not found")
    
    with tab_ref:
        st.markdown("### Reference Correction")
        render_reference_correction(sub_data['reference_correction'])
        
        if sub_data['assignment']:
            with st.expander("📋 Assignment (Consigna)"):
                st.markdown(sub_data['assignment'])
    
    with tab_compare:
        if not st.session_state.selected_runs:
            st.info("👈 Use the sidebar to add runs for comparison")
        else:
            # Sync mode checkbox at the top
            sync_col1, sync_col2 = st.columns([1, 5])
            with sync_col1:
                # Use a callback for more reliable state handling
                def on_sync_toggle():
                    new_state = st.session_state.sync_checkbox_key
                    if new_state != st.session_state.sync_enabled:
                        toggle_sync_mode(new_state)
                
                st.checkbox(
                    "🔗 Sync Compare", 
                    value=st.session_state.sync_enabled,
                    key="sync_checkbox_key",
                    on_change=on_sync_toggle,
                    help="When enabled, tab selection and expand/collapse actions will apply to all runs"
                )
            
            with sync_col2:
                if st.session_state.sync_enabled:
                    st.caption("🔗 Sync mode ON - Actions mirror across all runs")
            
            # When sync is enabled, show a shared tab selector
            if st.session_state.sync_enabled:
                tab_options = ["📝 Corrections", "📊 Metrics", "🎯 Evaluation", "🧠 Prompts"]
                
                def on_tab_change():
                    st.session_state.sync_active_tab = st.session_state.sync_tab_selector
                
                selected_tab = st.radio(
                    "Select View",
                    options=tab_options,
                    index=tab_options.index(st.session_state.sync_active_tab) if st.session_state.sync_active_tab in tab_options else 0,
                    key="sync_tab_selector",
                    on_change=on_tab_change,
                    horizontal=True,
                    label_visibility="collapsed"
                )
            
            st.divider()
            
            # Create columns for side-by-side comparison
            num_runs = len(st.session_state.selected_runs)
            cols = st.columns(num_runs)
            
            for i, run_key in enumerate(st.session_state.selected_runs):
                with cols[i]:
                    exp_type, timestamp = run_key.split('/')
                    run_data = get_run_data(config, exp_type, timestamp, submission_name)
                    render_run_data(run_data, run_key)


def main():
    """Main application entry point."""
    init_session_state()
    
    if st.session_state.current_page == 'home':
        render_home_page()
    elif st.session_state.current_page == 'submission':
        render_submission_page()


if __name__ == "__main__":
    main()
