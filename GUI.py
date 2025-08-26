import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go # Import graph_objects for 3D density and Mesh3d
import re  # for regex escaping
from collections import Counter # For counting top services
from sklearn.cluster import KMeans # For K-Means clustering
from scipy.spatial import ConvexHull # For drawing convex hulls

st.set_page_config(layout="wide", page_title="Service Embedding 3D Group Comparisons")

# --- Constants: Data file paths ---
CSV_EMB = 'embeddings_output_filtered.csv'
COMPANY_CSV = 'merged_filtered.csv'
SERVICE_COLS = [f'service_{i}' for i in range(1, 25)]

# --- Conceptual axes keywords ---
# NOTE: These keyword lists should be filled with actual relevant keywords for your domain.
# They define the 'direction' of your conceptual axes in the embedding space.
AXES = {
    'Ad/PR/Brand': ['advertising', 'marketing', 'public relations', 'brand strategy', 'campaign management', 'social media marketing'],
    'Media/Production': ['video production', 'audio production', 'content creation', 'broadcasting', 'film making', 'post-production'],
    'Design/UX/Web': ['web design', 'ui ux design', 'graphic design', 'product design', 'frontend development', 'user experience']
}
AXIS_NAMES = list(AXES.keys()) # Get names for interactive selection

# --- Sidebar: Display Options ---
st.sidebar.header("Display Options")
show_base = st.sidebar.checkbox("Show base layer (individual services)", value=True)

# --- Interactive Axis Selection ---
st.sidebar.subheader("3D Plot Axis Mapping")
selected_x_axis = st.sidebar.selectbox("X-Axis", AXIS_NAMES, index=0, key="x_axis_select")
selected_y_axis = st.sidebar.selectbox("Y-Axis", AXIS_NAMES, index=1, key="y_axis_select")
selected_z_axis = st.sidebar.selectbox("Z-Axis", AXIS_NAMES, index=2, key="z_axis_select")

# Ensure unique axes are selected
if len({selected_x_axis, selected_y_axis, selected_z_axis}) < 3:
    st.sidebar.warning("Please select three unique conceptual axes for X, Y, and Z.")
    st.stop()

# --- Metric Encoding Options ---
st.sidebar.subheader("Data Metrics Encoding")
metric_options = {
    "None": None,
    "Revenue": "revenue",
    "Headcount": "headcount",
    "Number of Services": "num_services_per_company" # Will be calculated
}
size_by_metric = st.sidebar.selectbox("Marker Size By", list(metric_options.keys()), index=0, key="size_metric")
color_by_metric = st.sidebar.selectbox("Marker Color By", list(metric_options.keys()), index=0, key="color_metric")

# --- K-Means Clustering Options ---
st.sidebar.subheader("Clustering")
enable_clustering = st.sidebar.checkbox("Enable K-Means Clustering", value=False)
n_clusters = 3 # Default
if enable_clustering:
    n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3, step=1)

# --- Group Visualization Options ---
st.sidebar.subheader("Group Visualization")
draw_convex_hulls = st.sidebar.checkbox("Draw Convex Hulls for Groups", value=False)


# --- Load embeddings ---
@st.cache_data
def load_embeddings():
    """
    Loads service embeddings from a CSV, parses JSON embeddings, and stacks them into a matrix.
    """
    try:
        services_df = pd.read_csv(CSV_EMB)
        # Ensure 'embedding' column is treated as strings before JSON parsing
        services_df['emb_vec'] = services_df['embedding'].apply(
            lambda e: np.array(json.loads(e)) if isinstance(e, str) else np.array(e)
        )
        emb_matrix = np.vstack(services_df['emb_vec'].values)
        return services_df, emb_matrix
    except FileNotFoundError:
        st.error(f"Error: {CSV_EMB} not found. Please ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        st.stop()

# --- Compute projection axes & services (enhanced to get top keywords) ---
@st.cache_data
def compute_projection(services_df, emb_matrix, selected_axes_order):
    """
    Computes projection axes based on keywords, projects service embeddings onto these axes,
    and identifies top services for each axis.
    """
    axis_vecs = {}
    axis_top_keywords = {} # Store top keywords for each axis

    for name in selected_axes_order:
        kws = AXES[name]
        if not kws:
            st.warning(f"No keywords provided for axis '{name}'. This axis may not be well-defined.")
            axis_vecs[name] = np.zeros(emb_matrix.shape[1])
            axis_top_keywords[name] = ["(No keywords defined)"]
            continue

        pattern = '|'.join([re.escape(kw) for kw in kws])
        mask = services_df['text'].str.contains(pattern, case=False, na=False)

        if mask.sum() == 0:
            st.warning(f"No services found matching keywords for axis '{name}'. Check keywords or data.")
            axis_vecs[name] = np.zeros(emb_matrix.shape[1])
            axis_top_keywords[name] = ["(No matching services)"]
        else:
            axis_emb_subset = services_df[mask.values].copy() # Ensure a copy to avoid SettingWithCopyWarning
            
            # Calculate mean vector for the axis
            axis_vecs[name] = np.vstack(axis_emb_subset['emb_vec'].values).mean(axis=0)

            # Calculate dot product of all services with this axis vector to find top terms
            # Need to re-apply dot product on `services_df` not `axis_emb_subset`
            all_services_with_dots = services_df.copy()
            all_services_with_dots['dot_product_with_axis'] = all_services_with_dots['emb_vec'].apply(lambda x: np.dot(x, axis_vecs[name]))

            # Get top services by dot product (positive correlation)
            top_services_for_axis = all_services_with_dots.sort_values(
                by='dot_product_with_axis', ascending=False
            ).head(10)['text'].tolist()
            axis_top_keywords[name] = top_services_for_axis

    A = np.vstack([axis_vecs[name] for name in selected_axes_order])
    Q, _ = np.linalg.qr(A.T)
    proj_matrix = Q.T

    proj = emb_matrix.dot(proj_matrix.T)
    services_df[['x', 'y', 'z']] = proj[:, :3]
    return services_df, proj_matrix, axis_top_keywords

# --- Load data ---
services_df, emb_matrix = load_embeddings()
services_df, proj_matrix, axis_top_keywords = compute_projection(services_df, emb_matrix, [selected_x_axis, selected_y_axis, selected_z_axis])

try:
    company_df = pd.read_csv(COMPANY_CSV)
    # Ensure year_founded is numeric, coercing errors to NaN
    company_df['year_founded'] = pd.to_numeric(company_df.get('year_founded'), errors='coerce')
    
    # Calculate number of services per company for metric encoding
    company_df['num_services_per_company'] = company_df[SERVICE_COLS].count(axis=1)

    # --- Robustly handle 'revenue' and 'headcount' columns ---
    # Check if 'revenue' column exists before processing
    if 'revenue' in company_df.columns:
        company_df['revenue'] = pd.to_numeric(company_df['revenue'], errors='coerce').fillna(0)
    else:
        company_df['revenue'] = 0 # If column doesn't exist, create it and fill with 0

    # Check if 'headcount' column exists before processing
    if 'headcount' in company_df.columns:
        company_df['headcount'] = pd.to_numeric(company_df['headcount'], errors='coerce').fillna(0)
    else:
        company_df['headcount'] = 0 # If column doesn't exist, create it and fill with 0
    # --- End of robust handling ---
    
    emb_map = dict(zip(services_df['text'], services_df['emb_vec']))
except FileNotFoundError:
    st.error(f"Error: {COMPANY_CSV} not found. Please ensure the file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading company data: {e}")
    st.stop()

# --- Display Axis Interpretation ---
st.sidebar.subheader("Axis Interpretation")
show_axis_keywords = st.sidebar.checkbox("Show Top Keywords for Axes", value=True) # Changed default to True
if show_axis_keywords:
    st.sidebar.markdown(f"**{selected_x_axis} (X-Axis):**")
    for kw in axis_top_keywords[selected_x_axis]:
        st.sidebar.markdown(f"- {kw}")
    st.sidebar.markdown(f"**{selected_y_axis} (Y-Axis):**")
    for kw in axis_top_keywords[selected_y_axis]:
        st.sidebar.markdown(f"- {kw}")
    st.sidebar.markdown(f"**{selected_z_axis} (Z-Axis):**")
    for kw in axis_top_keywords[selected_z_axis]:
        st.sidebar.markdown(f"- {kw}")

# --- Animation by Year Founded ---
st.sidebar.subheader("Animate by Year Founded")
# Filter out NaN years to determine min/max for the slider
valid_years = company_df['year_founded'].dropna()
min_year_all = int(valid_years.min()) if not valid_years.empty else 1900
max_year_all = int(valid_years.max()) if not valid_years.empty else 2024
selected_year_range_anim = st.sidebar.slider(
    "Select Year Range for Animation",
    min_value=min_year_all,
    max_value=max_year_all,
    value=(min_year_all, max_year_all),
    step=1,
    key="anim_year_range_slider" # Unique key for this slider
)
animate_by_year = st.sidebar.checkbox("Enable Year Animation", value=False)


# --- Initialize groups state ---
st.sidebar.header("Group Comparisons")
if 'groups' not in st.session_state:
    st.session_state.groups = []

# Function to add a new group to the session state
def add_group():
    st.session_state.groups.append({
        'name': f"New Group {len(st.session_state.groups) + 1}", # Default name
        'industry': ['All'],
        'city': ['All'],
        'country': ['All'],
        # Initialize group year filter with the full range of available years
        'year_range': (min_year_all, max_year_all), 
        'color': '#ff0000', # Default color (red)
        'density': False,
        'centroid': None, # To store calculated centroid
        'num_companies': 0, # To store number of companies in group
        'top_services': [] # To store top services for the group
    })

st.sidebar.button("Add Group", on_click=add_group)

# --- Build base 3D scatter figure ---
# Initialize the 3D plot
fig = px.scatter_3d()

# Add the base layer of individual services if toggled on
if show_base:
    base = px.scatter_3d(services_df, x='x', y='y', z='z', hover_name='text',
                         title="Service Embedding 3D Group Comparisons")
    # Update trace properties for base layer (smaller markers, higher opacity)
    for trace in base.data:
        trace.update(marker=dict(size=1.4, opacity=0.1), showlegend=False) # No legend for base layer
        fig.add_trace(trace)

# Update layout for the 3D scene (axis titles based on AXES keys)
fig.update_layout(
    scene=dict(
        xaxis_title=selected_x_axis,
        yaxis_title=selected_y_axis,
        zaxis_title=selected_z_axis,
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=1.5, z=1) # Fixed aspect ratio for consistent view
    ),
    margin=dict(l=0, r=0, t=30, b=0) # Adjust margins
)

# List to store centroids for pairwise distance calculation later
group_centroids = []

# --- Configure groups and overlay data ---
all_group_companies_df = pd.DataFrame() # To store data for animation frame and clustering

for idx, grp in enumerate(st.session_state.groups):
    # Defensive check: Ensure grp is a dictionary before accessing its keys
    if not isinstance(grp, dict):
        st.error(f"Invalid group data encountered at index {idx}. Skipping this entry.")
        continue
        
    # Create an expander for each group in the sidebar
    exp = st.sidebar.expander(f"Group {idx+1}: {grp.get('name', '')}")
    
    # Get unique filter options from company_df
    industries = ['All'] + sorted(company_df['industry'].dropna().unique().tolist())
    cities = ['All'] + sorted(company_df['headquarters_city'].dropna().unique().tolist())
    countries = ['All'] + sorted(company_df['headquarters_country'].dropna().unique().tolist())
    
    with exp:
        # Allow user to name the group
        grp['name'] = st.text_input("Group Name", value=grp.get('name', f"Group {idx+1}"), key=f"name_{idx}")
        # Multiselect filters for each group
        grp['industry'] = st.multiselect("Industry", industries, default=grp.get('industry', ['All']), key=f"industry_{idx}")
        grp['city'] = st.multiselect("Headquarters City", cities, default=grp.get('city', ['All']), key=f"city_{idx}")
        grp['country'] = st.multiselect("Headquarters Country", countries, default=grp.get('country', ['All']), key=f"country_{idx}")
        
        # Slider for group year filter
        grp['year_range'] = st.slider(
            "Year Founded (Group Filter)",
            min_value=min_year_all,
            max_value=max_year_all,
            value=grp.get('year_range', (min_year_all, max_year_all)), # Use default if not set
            step=1,
            key=f"group_year_slider_{idx}" # Unique key for each group's slider
        )
        
        grp['color'] = st.color_picker("Color", value=grp.get('color', '#ff0000'), key=f"color_{idx}")
        grp['density'] = st.checkbox("Show density (2D & 3D)", value=grp.get('density', False), key=f"density_{idx}")
        
        # Display centroid and number of companies for the group
        if grp.get('centroid') is not None and isinstance(grp['centroid'], np.ndarray) and len(grp['centroid']) >= 3:
            st.write(f"**Companies:** {grp['num_companies']}")
            st.write(f"**Centroid ({selected_x_axis}, {selected_y_axis}, {selected_z_axis}):** ({grp['centroid'][0]:.2f}, {grp['centroid'][1]:.2f}, {grp['centroid'][2]:.2f})")
        else:
            st.write(f"**Companies:** {grp['num_companies']}")
            st.write(f"**Centroid:** N/A")

        # Display Top Services for the group
        if grp['top_services']:
            st.write("**Top 5 Services:**")
            for service, count in grp['top_services']:
                st.write(f"- {service} ({count})")
        else:
            st.write("**Top Services:** N/A (no companies or services found)")

        # Button to remove the group
        if st.button("Remove Group", key=f"remove_{idx}"):
            st.session_state.groups.pop(idx)
            st.experimental_rerun()

    # Filter companies based on selected criteria for the current group
    mask = pd.Series(True, index=company_df.index)
    if 'All' not in grp.get('industry', ['All']): mask &= company_df['industry'].isin(grp['industry'])
    if 'All' not in grp.get('city', ['All']): mask &= company_df['headquarters_city'].isin(grp['city'])
    if 'All' not in grp.get('country', ['All']): mask &= company_df['headquarters_country'].isin(grp['country'])
    
    # Apply group's year filter using the slider range
    min_grp_year, max_grp_year = grp.get('year_range', (min_year_all, max_year_all))
    mask &= (company_df['year_founded'] >= min_grp_year) & (company_df['year_founded'] <= max_grp_year)
    
    current_group_companies_df = company_df[mask].copy()
    
    # Drop rows with NaN years at this point to prevent IntCastingNaNError
    current_group_companies_df = current_group_companies_df.dropna(subset=['year_founded'])

    # Apply animation year filter if enabled
    if animate_by_year:
        current_group_companies_df = current_group_companies_df[
            (current_group_companies_df['year_founded'] >= selected_year_range_anim[0]) &
            (current_group_companies_df['year_founded'] <= selected_year_range_anim[1])
        ]
    
    # Prepare embeddings and collect services for the current group
    emb_list, labels, all_services_in_group, years_for_proj, \
    company_revenues, company_headcounts, company_num_services = [], [], [], [], [], [], [] 

    for i in current_group_companies_df.index:
        row = current_group_companies_df.loc[i]
        company_services = [row[c] for c in SERVICE_COLS if pd.notna(row[c])]
        all_services_in_group.extend(company_services)

        vecs = [emb_map.get(s) for s in company_services]
        vecs = [v for v in vecs if v is not None]
        
        if vecs:
            emb_list.append(np.mean(vecs, axis=0))
            
            # Enhanced label with all available metrics
            label_str = (
                f"Company: {row.get('name', 'N/A')}<br>"
                f"Industry: {row.get('industry', 'N/A')}<br>"
                f"City: {row.get('headquarters_city', 'N/A')}<br>"
                f"Country: {row.get('headquarters_country', 'N/A')}<br>"
                f"Year Founded: {int(row.get('year_founded')) if pd.notna(row.get('year_founded')) else 'N/A'}"
            )
            # Only add if actual value exists and > 0
            if pd.notna(row.get('revenue')) and row.get('revenue') > 0: 
                label_str += f"<br>Revenue: ${row['revenue']:,}"
            # Only add if actual value exists and > 0
            if pd.notna(row.get('headcount')) and row.get('headcount') > 0: 
                label_str += f"<br>Headcount: {int(row['headcount']):,}"
            # Only add if actual value exists and > 0
            if pd.notna(row.get('num_services_per_company')) and row.get('num_services_per_company') > 0: 
                label_str += f"<br>Services Offered: {int(row['num_services_per_company'])}"
            
            labels.append(label_str)
            years_for_proj.append(int(row['year_founded'])) 
            company_revenues.append(row['revenue'])
            company_headcounts.append(row['headcount'])
            company_num_services.append(row['num_services_per_company'])
    
    grp['num_companies'] = len(current_group_companies_df) # Store number of companies
    
    # Calculate Top Services for the group
    service_counts = Counter(all_services_in_group)
    grp['top_services'] = service_counts.most_common(5) # Store top 5 services

    if not emb_list:
        st.sidebar.warning(f"Group {idx+1}: No companies found for the selected filters and year range.")
        grp['centroid'] = None
        continue

    proj_comp = np.vstack(emb_list).dot(proj_matrix.T)
    comp_df = pd.DataFrame(proj_comp, columns=['x', 'y', 'z'])
    comp_df['label'] = labels
    comp_df['group_name'] = grp['name'] # For animation group and group coloring
    comp_df['group_color'] = grp['color'] # Store the group's color
    
    comp_df['year_founded_for_anim'] = years_for_proj 
    comp_df['revenue'] = company_revenues
    comp_df['headcount'] = company_headcounts
    comp_df['num_services_per_company'] = company_num_services
    
    grp['centroid'] = comp_df[['x', 'y', 'z']].mean().values
    # Store group color along with centroid for convex hull drawing
    group_centroids.append({'name': grp['name'], 'centroid': grp['centroid'], 'color': grp['color']})

    # Concatenate this group's data for potential animation and global clustering
    all_group_companies_df = pd.concat([all_group_companies_df, comp_df], ignore_index=True)

    # Add scatter plot for the current group if not animating and not clustering globally
    if not animate_by_year and not enable_clustering:
        marker_size_col = metric_options[size_by_metric]
        marker_color_col = metric_options[color_by_metric]

        marker_dict = dict(size=2, color=grp['color'], opacity=0.7) # Default to group color
        if marker_size_col:
            # Ensure sizes are non-negative and handle potential NaNs (though filled earlier)
            marker_dict['size'] = comp_df[marker_size_col].replace([np.inf, -np.inf], np.nan).fillna(0).values
            # Scale sizes for better visibility, e.g., using sizemode and sizeref
            marker_dict['sizemode'] = 'area' # 'area' or 'diameter'
            if comp_df[marker_size_col].max() > 0:
                marker_dict['sizeref'] = comp_df[marker_size_col].max() / 50 # Adjust ref for reasonable size
            else:
                marker_dict['sizeref'] = 1 # Avoid division by zero
            marker_dict['sizemin'] = 3 # Minimum size for very small values
            
        if marker_color_col:
            # Override group color if a metric is chosen for coloring
            marker_dict['color'] = comp_df[marker_color_col]
            marker_dict['colorscale'] = 'Viridis' # Choose a continuous color scale
            marker_dict['colorbar'] = dict(title=marker_color_col.replace('_', ' ').title(), len=0.5) # Add colorbar
            
        fig.add_scatter3d(x=comp_df['x'], y=comp_df['y'], z=comp_df['z'],
                          mode='markers', text=comp_df['label'], hoverinfo='text',
                          marker=marker_dict, name=grp['name'],
                          legendgroup=grp['name'], showlegend=True) # Ensure legend shows for groups
    # Add 2D density contours and 3D volume if 'density' is checked and not animating/clustering
    if grp.get('density') and not animate_by_year and not enable_clustering:
        # Generate separate 2D density charts
        if grp.get('density'): # Only show if enabled for the group
            st.markdown(f"---")
            st.subheader(f"Density Plots for {grp['name']}")
            planes = [('x','y', selected_x_axis, selected_y_axis), 
                      ('y','z', selected_y_axis, selected_z_axis), 
                      ('x','z', selected_x_axis, selected_z_axis)]
            for a_col, b_col, a_title, b_title in planes:
                dens2d = px.density_contour(comp_df, x=a_col, y=b_col,
                                            title=f"Density 2D ({a_title} vs {b_title}) - {grp['name']}",
                                            labels={a_col: a_title, b_col: b_title},
                                            color_discrete_sequence=[grp['group_color']]) # Use group color
                dens2d.update_traces(ncontours=10, selector=dict(type='contour'))
                dens2d.update_layout(coloraxis_showscale=False, showlegend=False)
                st.plotly_chart(dens2d, use_container_width=True)
            
            # Add 3D density (volume) map using plotly.graph_objects
            data_points = comp_df[['x', 'y', 'z']].values
            
            if not comp_df.empty:
                x_min, x_max = comp_df['x'].min(), comp_df['x'].max()
                y_min, y_max = comp_df['y'].min(), comp_df['y'].max()
                z_min, z_max = comp_df['z'].min(), comp_df['z'].max()

                n_bins = 10 # Number of bins along each axis for density estimation
                # Add a small epsilon to range to avoid issues with single point ranges
                x_range = [x_min - 0.01, x_max + 0.01] if x_min == x_max else [x_min, x_max]
                y_range = [y_min - 0.01, y_max + 0.01] if y_min == y_max else [y_min, y_max]
                z_range = [z_min - 0.01, z_max + 0.01] if z_min == z_max else [z_min, z_max]

                hist, edges = np.histogramdd(data_points, bins=n_bins, range=[x_range, y_range, z_range])

                X_hist = (edges[0][:-1] + edges[0][1:]) / 2
                Y_hist = (edges[1][:-1] + edges[1][1:]) / 2
                Z_hist = (edges[2][:-1] + edges[2][1:]) / 2

                fig.add_trace(go.Volume(
                    x=X_hist,
                    y=Y_hist,
                    z=Z_hist,
                    value=hist.flatten(),
                    isomin=hist.max() * 0.1, # Start rendering from 10% of max density for visibility
                    isomax=hist.max(),
                    opacity=0.15,
                    surface_count=5, # Number of iso-surfaces
                    colorscale=[[0, grp['group_color']], [1, grp['group_color']]], # Use group color
                    name=f"Density 3D ({grp['name']})",
                    showscale=False,
                    hoverinfo='name'
                ))
            else:
                st.info(f"Not enough data points to compute 3D density for {grp['name']}.")


# --- Apply Clustering if enabled ---
if enable_clustering and not all_group_companies_df.empty:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    all_group_companies_df['cluster'] = kmeans.fit_predict(all_group_companies_df[['x', 'y', 'z']])
    
    # Determine size and color arguments based on user selection
    marker_size_col = metric_options[size_by_metric]
    marker_color_col = metric_options[color_by_metric]

    fig = px.scatter_3d(
        all_group_companies_df,
        x='x', y='y', z='z',
        color=marker_color_col if marker_color_col else 'cluster', # Color by metric or cluster
        color_continuous_scale='Viridis' if marker_color_col else px.colors.qualitative.Plotly,
        color_discrete_sequence=None if marker_color_col else px.colors.qualitative.Plotly, # Use qualitative for clusters
        size=marker_size_col,
        size_max=50, # Max size for markers
        hover_name='label',
        title=f"Service Embedding 3D Group Comparisons with {n_clusters} Clusters"
    )
    # Restore base layer if needed
    if show_base:
        for trace in base.data:
            fig.add_trace(trace)
    # Restore layout settings for axis titles
    fig.update_layout(
        scene=dict(
            xaxis_title=selected_x_axis,
            yaxis_title=selected_y_axis,
            zaxis_title=selected_z_axis,
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

# --- Draw Convex Hulls if enabled and not animating ---
if draw_convex_hulls and not animate_by_year:
    st.subheader("Convex Hulls")
    for grp_info in group_centroids: # Iterate through collected group info (with color)
        # Filter for the specific group's data from the consolidated DataFrame
        current_group_data_for_hull = all_group_companies_df[all_group_companies_df['group_name'] == grp_info['name']]

        if len(current_group_data_for_hull) >= 4: # Need at least 4 points for a 3D hull
            points = current_group_data_for_hull[['x', 'y', 'z']].values
            try:
                hull = ConvexHull(points)
                # Create a Mesh3d trace for the convex hull
                fig.add_trace(go.Mesh3d(
                    x=points[:,0], # All x coordinates
                    y=points[:,1], # All y coordinates
                    z=points[:,2], # All z coordinates
                    i=hull.simplices[:,0], # Index of the first vertex of triangles
                    j=hull.simplices[:,1], # Index of the second vertex of triangles
                    k=hull.simplices[:,2], # Index of the third vertex of triangles
                    opacity=0.15,
                    color=grp_info['color'], # Use the group's color
                    name=f"Hull: {grp_info['name']}",
                    showlegend=True,
                    hoverinfo='name',
                    flatshading=True # Use flat shading for a faceted look
                ))
            except Exception as e:
                st.warning(f"Could not draw convex hull for {grp_info['name']}: {e}. Ensure enough distinct points.")
        else:
            st.info(f"Not enough points to draw convex hull for '{grp_info['name']}'. Requires at least 4 non-coplanar points.")


# --- Render 3D scatter plot ---
# Conditional rendering for animation or static plot
if animate_by_year and not all_group_companies_df.empty:
    # Sort the DataFrame by the animation frame column to ensure chronological order
    all_group_companies_df.sort_values(by='year_founded_for_anim', inplace=True)

    marker_size_anim_arg = metric_options[size_by_metric]
    marker_color_anim_arg = metric_options[color_by_metric]

    animated_fig = px.scatter_3d(
        all_group_companies_df,
        x='x', y='y', z='z',
        animation_frame='year_founded_for_anim', # Animate by year
        animation_group='label', # Group by company for consistent tracking
        color=marker_color_anim_arg if marker_color_anim_arg else 'group_name',
        color_discrete_map={g['name']: g['color'] for g in st.session_state.groups if g['name'] is not None} if not marker_color_anim_arg else None,
        color_continuous_scale='Viridis' if marker_color_anim_arg else None,
        size=marker_size_anim_arg,
        size_max=50, # Max size for animation markers
        hover_name='label',
        range_x=[all_group_companies_df['x'].min(), all_group_companies_df['x'].max()],
        range_y=[all_group_companies_df['y'].min(), all_group_companies_df['y'].max()],
        range_z=[all_group_companies_df['z'].min(), all_group_companies_df['z'].max()],
        title=f"Service Embedding 3D Group Comparisons (Animated by Year: {selected_year_range_anim[0]}-{selected_year_range_anim[1]})"
    )
    # Update layout for the animated figure
    animated_fig.update_layout(
        scene=dict(
            xaxis_title=selected_x_axis,
            yaxis_title=selected_y_axis,
            zaxis_title=selected_z_axis,
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[dict(type='buttons',
                          showactive=True,
                          buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, {'frame': {'duration': 500, 'redraw': True},
                                                      'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}])])],
        sliders=[dict(steps=[dict(args=[[f'{s.name}'], {'frame': {'duration': 500, 'redraw': True},
                                                                           'mode': 'immediate',
                                                                           'transition': {'duration': 300}}],
                                  label=str(s.name),
                                  method='animate') for s in animated_fig.frames])]
    )
    # Add base layer to animated figure if requested
    if show_base:
        for trace in base.data:
            animated_fig.add_trace(trace)
    
    st.plotly_chart(animated_fig, use_container_width=True)

else: # If not animating, display the static plot (potentially with clustering/hulls)
    st.plotly_chart(fig, use_container_width=True)

# --- Additional Views and Summaries ---

## Pairwise Centroid Distances
# If there's more than one group defined, this section in the sidebar displays the Euclidean distances
# between the centroids of each pair of groups, quantifying their similarity in the service space.
if len(group_centroids) > 1:
    st.sidebar.subheader("Group Centroid Distances")
    for i in range(len(group_centroids)):
        for j in range(i + 1, len(group_centroids)):
            g1_name = group_centroids[i]['name']
            g2_name = group_centroids[j]['name']
            g1_centroid = group_centroids[i]['centroid']
            g2_centroid = group_centroids[j]['centroid']
            
            distance = np.linalg.norm(g1_centroid - g2_centroid)
            st.sidebar.write(f"**{g1_name}** to **{g2_name}**: {distance:.2f}")

## 2D Scatterplot Matrix
# This view provides all three pairwise 2D scatter plots, allowing for easier comparison of density and
# outliers across different conceptual axis pairings.
st.subheader("2D Paired Views")
show_scatterplot_matrix = st.checkbox("Show 2D Scatterplot Matrix", value=False)

if show_scatterplot_matrix and not all_group_companies_df.empty:
    # Filter for relevant columns for the matrix
    cols_for_matrix = ['x', 'y', 'z', 'group_name', 'label']
    if size_by_metric and metric_options[size_by_metric] and metric_options[size_by_metric] in all_group_companies_df.columns:
        cols_for_matrix.append(metric_options[size_by_metric])
    if color_by_metric and metric_options[color_by_metric] and metric_options[color_by_metric] in all_group_companies_df.columns:
        cols_for_matrix.append(metric_options[color_by_metric])
    elif enable_clustering: # If clustering is enabled, 'cluster' column is present
        cols_for_matrix.append('cluster')

    # Ensure all columns in cols_for_matrix exist in all_group_companies_df
    cols_for_matrix = [col for col in cols_for_matrix if col in all_group_companies_df.columns]

    scatter_matrix_fig = px.scatter_matrix(
        all_group_companies_df[cols_for_matrix],
        dimensions=['x', 'y', 'z'],
        color=color_by_metric if color_by_metric and metric_options[color_by_metric] else ('cluster' if enable_clustering else 'group_name'),
        color_discrete_map={g['name']: g['color'] for g in st.session_state.groups if g['name'] is not None} if not color_by_metric and not enable_clustering else None,
        color_continuous_scale='Viridis' if color_by_metric else None,
        hover_name='label',
        title="2D Scatterplot Matrix of Groups"
    )
    # Update axis labels to match conceptual axes
    axis_mapping = {'x': selected_x_axis, 'y': selected_y_axis, 'z': selected_z_axis}
    for i, axis_col in enumerate(['x', 'y', 'z']):
        scatter_matrix_fig.update_layout({
            f'xaxis{i+1}_title': axis_mapping.get(axis_col, axis_col),
            f'yaxis{i+1}_title': axis_mapping.get(axis_col, axis_col),
        })
    st.plotly_chart(scatter_matrix_fig, use_container_width=True)

## Group Summary Statistics
# This table provides a quick overview of each defined group, including their central tendencies (centroids),
# the number of companies they contain, and their most common service offerings.
st.subheader("Group Summary Statistics")
if group_centroids:
    summary_data = []
    for grp_info in st.session_state.groups: # Iterate through the original group state to get all info
        if grp_info['centroid'] is not None:
            summary_data.append({
                "Group Name": grp_info['name'],
                "Companies": grp_info['num_companies'],
                f"{selected_x_axis} Centroid": f"{grp_info['centroid'][0]:.2f}",
                f"{selected_y_axis} Centroid": f"{grp_info['centroid'][1]:.2f}",
                f"{selected_z_axis} Centroid": f"{grp_info['centroid'][2]:.2f}",
                "Top Services": ", ".join([s[0] for s in grp_info['top_services']])
            })
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No group data to display summary statistics.")
else:
    st.info("Add groups to see summary statistics.")

## Company Details Drill-down
# Select specific companies from the dropdown to view their detailed profiles in a tabular format,
# including all their identified services.
st.subheader("Company Details Drill-down")
all_company_names = sorted(company_df['name'].dropna().unique().tolist())
selected_companies_for_detail = st.multiselect(
    "Select Companies to View Details",
    all_company_names,
    key="company_detail_select"
)

if selected_companies_for_detail:
    detailed_df = company_df[company_df['name'].isin(selected_companies_for_detail)].copy()
    
    # Format service columns for better display
    detailed_df['Services'] = detailed_df.apply(
        lambda row: ", ".join([str(row[c]) for c in SERVICE_COLS if pd.notna(row[c])]), axis=1
    )
    
    # Select and display relevant columns for detail
    display_cols = ['name', 'industry', 'headquarters_city', 'headquarters_country', 
                    'year_founded', 'revenue', 'headcount', 'num_services_per_company', 'Services']
    
    # Filter display_cols to only include columns that actually exist in detailed_df
    display_cols = [col for col in display_cols if col in detailed_df.columns]
    
    st.dataframe(detailed_df[display_cols].fillna('N/A'), use_container_width=True)

