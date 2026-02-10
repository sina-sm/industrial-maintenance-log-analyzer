import streamlit as st
import pandas as pd
import spacy
import plotly.express as px
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np

# -------------------------------
# Initial Setup for Streamlit
# -------------------------------

# Set up the Streamlit page layout and title
st.set_page_config(page_title="Industrial NLP Analyzer", layout="wide")

# Load NLP model (spaCy) for text processing
@st.cache_resource
def load_model():
    """
    This function loads the pre-trained spaCy model to process text for maintenance logs.
    It ensures that the model is ready before use, otherwise, it will show an error message.
    """
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.error("Model not found. Please run: python -m spacy download en_core_web_sm")
        return None

# Load the model for text processing
nlp = load_model()

# Display project title and description
st.title("Industrial Maintenance Log Analyzer")
st.markdown("### AI-Driven Insights for Maintenance Logs")

# -------------------------------
# Load Data from Excel
# -------------------------------

# Attempt to load data from the 'data.xlsx' file
try:
    df = pd.read_excel("data.xlsx")
except FileNotFoundError:
    st.error("File not found! Please ensure 'data.xlsx' is placed in the same directory.")
    st.stop()

# Define failure categories for classification
LABELS = [
    "Leak",
    "Overheat",
    "Jam",
    "Electrical/Start_Failure",
    "Wear/Tear",
    "Noise/Vibration",
    "Misalignment",
    "General/Other",
]

# -------------------------------
# Helper Function: Label Suggestion
# -------------------------------

def suggest_label(text: str) -> str:

    if not isinstance(text, str) or not text.strip():
        return "General/Other"

    t = text.lower()

    # Define regex patterns for various fault types
    # FIX: Added parentheses () to group keywords so \b applies to all of them.
    # This prevents partial matches like 'coil' triggering 'oil' (Leak).
    rules = [
        ("Leak", r"\b(leak|leaking|fluid|oil|coolant)\b"),
        ("Overheat", r"\b(overheat|overheated|temperature|hot)\b"),
        ("Jam", r"\b(jam|stuck|blocked|clog)\b"),
        ("Electrical/Start_Failure", r"\b(trip|tripped|breaker|electrical|short|start fail|won't start|power|fuse|fused)\b"),
        ("Wear/Tear", r"\b(wear|worn|tear|torn|abrasion|corrosion|eroded)\b"),
        ("Noise/Vibration", r"\b(noise|noisy|vibration|vibrate|rattle|grinding|knocking)\b"),
        ("Misalignment", r"\b(misalign|misalignment|alignment|offset)\b"),
    ]

    # Apply pattern matching to suggest a label
    for label, pattern in rules:
        if re.search(pattern, t):
            return label

    return "General/Other"

# -------------------------------
# NLP Extraction: Failure & Component Detection
# -------------------------------

def extract_maintenance_info(log_text):
    if not isinstance(log_text, str):
        return pd.Series(["Unknown", "Unknown", "Unknown"])

    # Process the log text with spaCy
    doc = nlp(log_text.lower())

    # ------------------------
    # 1. Failure Mode Detection
    # ------------------------
    failure_keywords = ["leak", "fail", "overheat", "noise", "vibration", "jam", "break", "tear", "wear", "trip", "stop", "drop", "slip", "cut", "damage"]
    detected_failure = "General Fault"

    # Detect failure mode using the defined keywords
    for token in doc:
        if any(keyword in token.lemma_ for keyword in failure_keywords):
            detected_failure = token.text
            break

    # ------------------------
    # 2. Action & Component Extraction
    # ------------------------
    action = "Inspect/Check"
    target_component = "General Component"

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
            action = token.text
            for child in token.children:
                if child.dep_ in ["dobj", "pobj"]:
                    target_component = child.text
                    for grandchild in child.children:
                        if grandchild.dep_ in ["amod", "compound"]:
                            target_component = f"{grandchild.text} {target_component}"

    return pd.Series([detected_failure, action, target_component])

# -------------------------------
# Sidebar Filter for Machine Type Selection
# -------------------------------

# Sidebar for filtering logs based on machine type
st.sidebar.header("Filter Options")
if "Machine_Type" in df.columns:
    machines = df["Machine_Type"].unique()
    selected_machine = st.sidebar.multiselect("Select Machine", machines, default=list(machines))
    filtered_df = df[df["Machine_Type"].isin(selected_machine)].copy()
else:
    st.error("Column 'Machine_Type' not found in the Excel file!")
    st.stop()

# -------------------------------
# Analysis & Labeling Tabs
# -------------------------------

tab1, tab2, tab3 = st.tabs(["Analyze (Rule-based)", "Label Data", "Train & Evaluate"])

# -------------------------------
# Tab 1: Rule-based Analysis
# -------------------------------

with tab1:
    st.subheader("Rule-based NLP Analysis (Baseline)")
    st.markdown("This section extracts insights using predefined linguistic rules and keywords.")

    if st.button("Analyze Logs with AI (Rule-based)"):
        with st.spinner('Processing logs...'):
            if "Log_Message" in filtered_df.columns:
                # Extract key information from logs
                filtered_df[["Detected_Failure", "Action_Taken", "Component"]] = filtered_df["Log_Message"].apply(extract_maintenance_info)

                # Display KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Total Logs Analyzed", len(filtered_df))
                kpi2.metric("Top Failure Mode", filtered_df["Detected_Failure"].mode()[0] if not filtered_df.empty else "N/A")
                kpi3.metric("Most Replaced Part", filtered_df["Component"].mode()[0] if not filtered_df.empty else "N/A")

                # Display Charts
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Failure Modes Distribution")
                    fig1 = px.bar(filtered_df['Detected_Failure'].value_counts(), orientation='h', title="Frequency of Detected Failures")
                    st.plotly_chart(fig1, use_container_width=True)
                with c2:
                    st.subheader("Machine Breakdown")
                    fig2 = px.pie(filtered_df, names='Machine_Type', hole=0.4, title="Reports per Machine Type")
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Processed Data Table")
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.error("Column 'Log_Message' not found! Please check the input file headers.")
# -------------------------------
# Tab 2: Data Labeling (Human-in-the-loop)
# -------------------------------

with tab2:
    st.subheader("Step 1 — Create Labels (Data Annotation)")
    st.markdown("Review the AI-suggested labels and correct them to create a Ground Truth dataset.")

    if "Log_Message" not in filtered_df.columns:
        st.error("Column 'Log_Message' not found in the file.")
        st.stop()

    # Prepare a working copy of the filtered DataFrame
    work_df = filtered_df.copy()

    if "Failure_Category" not in work_df.columns:
        work_df["Failure_Category"] = ""

    # Generate suggestions for labels based on the log messages
    if "Suggested_Label" not in work_df.columns:
        work_df["Suggested_Label"] = work_df["Log_Message"].apply(suggest_label)

    # Prefill empty labels with suggestions
    work_df["Failure_Category"] = work_df["Failure_Category"].fillna("")
    mask_empty = work_df["Failure_Category"].astype(str).str.strip() == ""
    work_df.loc[mask_empty, "Failure_Category"] = work_df.loc[mask_empty, "Suggested_Label"]

    # Flag rows that need review based on AI suggestions
    if "Needs_Review" not in work_df.columns:
        work_df["Needs_Review"] = work_df["Suggested_Label"].eq("General/Other")

    st.caption("Automated suggestions applied. Please review and edit before training.")

    # Data Editor for manual correction of labels
    edited = st.data_editor(
        work_df[["Machine_Type", "Log_Message", "Suggested_Label", "Failure_Category", "Needs_Review"]],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Failure_Category": st.column_config.SelectboxColumn(
                "Failure_Category",
                help="Select the correct failure category",
                options=LABELS,
                required=True,
            ),
            "Needs_Review": st.column_config.CheckboxColumn(
                "Needs_Review",
                help="Check if this log is ambiguous",
            ),
        },
        hide_index=True,
    )

    # Buttons for saving labeled data
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Save labeled_data.xlsx"):
            # Prepare output DataFrame with labeled data
            out = filtered_df.copy()
            # Update with edited values
            out.loc[edited.index, "Suggested_Label"] = edited["Suggested_Label"].values
            out.loc[edited.index, "Failure_Category"] = edited["Failure_Category"].values
            out.loc[edited.index, "Needs_Review"] = edited["Needs_Review"].values

            out.to_excel("labeled_data.xlsx", index=False)
            st.success("'labeled_data.xlsx' saved successfully!")

    with colB:
        st.info("Go to the 'Train & Evaluate' tab to build the ML model.")

# -------------------------------
# Tab 3: Train & Evaluate Machine Learning Model
# -------------------------------

with tab3:
    st.subheader("Step 2 — Train ML Model & Evaluate")
    st.markdown("Train a Logistic Regression model on your labeled data and compare it with the rule-based baseline.")

    # Try to load the labeled data for training
    try:
        labeled_df = pd.read_excel("labeled_data.xlsx")
    except FileNotFoundError:
        st.error("'labeled_data.xlsx' not found! Please save your labels in the 'Label Data' tab first.")
        st.stop()

    required_cols = {"Log_Message", "Failure_Category"}
    if not required_cols.issubset(set(labeled_df.columns)):
        st.error("Required columns missing: 'Log_Message' and 'Failure_Category'.")
        st.stop()

    # Data Cleaning
    labeled_df = labeled_df.copy()
    labeled_df["Log_Message"] = labeled_df["Log_Message"].fillna("").astype(str)
    labeled_df["Failure_Category"] = labeled_df["Failure_Category"].fillna("").astype(str)

    # Filter out empty rows
    labeled_df = labeled_df[labeled_df["Failure_Category"].str.strip() != ""]
    labeled_df = labeled_df[labeled_df["Log_Message"].str.strip() != ""]

    if labeled_df.empty:
        st.error("No labeled rows found. Please return to the previous tab and label your data.")
        st.stop()

    if len(labeled_df) < 2:
        st.error("Insufficient data (less than 2 rows). Cannot perform Train/Test Split.")
        st.stop()

    st.write(f"Total Labeled Samples: **{len(labeled_df)}**")

    # Apply Sidebar Machine Filter to Training Data
    if "Machine_Type" in labeled_df.columns:
        labeled_df["Machine_Type"] = labeled_df["Machine_Type"].fillna("").astype(str).str.strip()
        selected_machine_norm = [str(m).strip() for m in selected_machine]
        labeled_df = labeled_df[labeled_df["Machine_Type"].isin(selected_machine_norm)].copy()

        if labeled_df.empty:
            st.error("No labeled data remaining after applying machine filters. Please broaden your selection.")
            st.stop()

    # Display Class Distribution Chart
    st.markdown("### Class Distribution")
    n_classes = labeled_df["Failure_Category"].nunique()
    if n_classes < 2:
        st.warning("Warning: Only one class detected. Classification requires at least 2 classes.")

    dist = labeled_df["Failure_Category"].value_counts().reset_index()
    dist.columns = ["Failure_Category", "count"]
    st.plotly_chart(px.bar(dist, x="Failure_Category", y="count"), use_container_width=True)

    # Train/Test Split
    X = labeled_df["Log_Message"].values
    y = labeled_df["Failure_Category"].values

    test_size = st.slider("Test Size Ratio", min_value=0.2, max_value=0.4, value=0.2, step=0.05)

    use_stratify = labeled_df["Failure_Category"].nunique() > 1
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if use_stratify else None
        )
        stratify_msg = "Stratified split successful."
    except ValueError:
        # Fallback if stratify fails due to small class counts
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        stratify_msg = "Stratified split failed (small class size). Used random split."

    st.caption(stratify_msg)

    # Model Pipeline Configuration
    C_param = st.slider("Regularization Strength (C)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)),
        ("lr", LogisticRegression(max_iter=2000, C=C_param, class_weight="balanced"))
    ])

    if st.button("Train & Evaluate Model"):
        with st.spinner("Training model..."):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # Display Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        k1, k2, k3 = st.columns(3)
        k1.metric("Accuracy", f"{acc:.3f}")
        k2.metric("F1-Score (Macro)", f"{f1m:.3f}")
        k3.metric("Test Samples", len(y_test))

        st.markdown("### Classification Report")
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).transpose()
        st.dataframe(rep_df, use_container_width=True)

        st.markdown("### Confusion Matrix")
        labels_sorted = sorted(list(set(y)))
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
        st.dataframe(cm_df, use_container_width=True)

        # ---- Comparison Section ----
        st.markdown("## Model vs. Rule-Based Baseline")

        # Generate Rule-based predictions for the test set
        y_rule = np.array([suggest_label(t) for t in X_test])
        # Map unknown rules to 'General/Other' if they aren't in the official list
        y_rule = np.array([lbl if lbl in LABELS else "General/Other" for lbl in y_rule])

        f1_rule = f1_score(y_test, y_rule, average="macro")
        acc_rule = accuracy_score(y_test, y_rule)

        comp = pd.DataFrame({
            "Approach": ["Rule-based (Keywords)", "Machine Learning (TF-IDF + LR)"],
            "Accuracy": [acc_rule, acc],
            "F1-Score (Macro)": [f1_rule, f1m],
        })
        st.dataframe(comp, use_container_width=True)

        # Error Analysis
        st.markdown("### Error Analysis (Misclassified Samples)")
        wrong_idx = np.where(y_pred != y_test)[0]
        if len(wrong_idx) == 0:
            st.success("No misclassifications in the test set!")
        else:
            show_n = min(10, len(wrong_idx))
            rows = []
            for i in wrong_idx[:show_n]:
                rows.append({
                    "Log Message": X_test[i],
                    "True Label": y_test[i],
                    "ML Prediction": y_pred[i],
                    "Rule Prediction": y_rule[i],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
