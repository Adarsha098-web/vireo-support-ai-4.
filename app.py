import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Vireo Support AI",
    page_icon="🎧",
    layout="wide"
)

@st.cache_data
def load_data():
    tickets = pd.read_csv("data/tickets.csv")
    agents = pd.read_csv("data/agents.csv")

    for col in ["created_at", "first_response_at", "resolved_at"]:
        if col in tickets.columns:
            tickets[col] = pd.to_datetime(tickets[col], errors="coerce")

    for col in ["from_date", "to_date"]:
        if col in agents.columns:
            agents[col] = pd.to_datetime(agents[col], errors="coerce")

    tickets["customer_message"] = tickets["customer_message"].fillna("")
    tickets["agent_notes"] = tickets["agent_notes"].fillna("")
    tickets["text"] = (
        tickets["customer_message"].astype(str)
        + " "
        + tickets["agent_notes"].astype(str)
    )

    start_date = pd.Timestamp("2025-01-01")
    end_date = pd.Timestamp("2026-07-01")

    tickets = tickets[
        (tickets["created_at"] >= start_date)
        & (tickets["created_at"] < end_date)
    ].copy()

    return tickets, agents


@st.cache_resource
def train_model(texts, labels):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def get_resolved_team(row, agents):
    if pd.isna(row.get("agent_id")) or pd.isna(row.get("resolved_at")):
        return None

    matches = agents[
        agents["agent_id"].astype(str) == str(row["agent_id"])
    ].copy()

    if matches.empty:
        return None

    resolved_at = row["resolved_at"]

    valid = matches[
        (
            matches["from_date"].isna()
            | (matches["from_date"] <= resolved_at)
        )
        & (
            matches["to_date"].isna()
            | (matches["to_date"] >= resolved_at)
        )
    ]

    if valid.empty:
        return None

    valid = valid.sort_values("from_date", ascending=False)

    return valid.iloc[0]["team"]


tickets, agents = load_data()

st.title("Vireo Support AI")
st.caption("AI-assisted ticket categorization and support routing analysis")

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tickets", f"{len(tickets):,}")

with col2:
    st.metric("Support Teams", tickets["assigned_team"].nunique())

with col3:
    st.metric("Categories", tickets["category"].nunique())

with col4:
    st.metric(
        "Avg Transfers",
        f"{tickets['transfers'].fillna(0).mean():.2f}"
    )

st.divider()

st.subheader("Monthly Ticket Volume")

monthly = (
    tickets.assign(month=tickets["created_at"].dt.to_period("M").astype(str))
    .groupby("month")
    .size()
    .reset_index(name="tickets")
)

fig_monthly = px.line(
    monthly,
    x="month",
    y="tickets",
    markers=True
)

fig_monthly.update_layout(
    xaxis_title="Month",
    yaxis_title="Tickets",
    height=420
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True,
    key="monthly_volume"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ticket Volume by Category")

    category_data = (
        tickets["category"]
        .value_counts()
        .reset_index()
    )

    category_data.columns = ["category", "tickets"]

    fig_category = px.bar(
        category_data,
        x="tickets",
        y="category",
        orientation="h"
    )

    fig_category.update_layout(
        height=500,
        yaxis_title="Category",
        xaxis_title="Tickets"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True,
        key="category_volume"
    )

with col2:
    st.subheader("Ticket Volume by Team")

    team_data = (
        tickets["assigned_team"]
        .value_counts()
        .reset_index()
    )

    team_data.columns = ["team", "tickets"]

    fig_team = px.bar(
        team_data,
        x="tickets",
        y="team",
        orientation="h"
    )

    fig_team.update_layout(
        height=500,
        yaxis_title="Team",
        xaxis_title="Tickets"
    )

    st.plotly_chart(
        fig_team,
        use_container_width=True,
        key="team_volume"
    )

st.divider()

st.subheader("Monthly Category Breakdown")

monthly_category = (
    tickets.assign(
        month=tickets["created_at"].dt.to_period("M").astype(str)
    )
    .groupby(["month", "category"])
    .size()
    .reset_index(name="tickets")
)

fig_monthly_category = px.bar(
    monthly_category,
    x="month",
    y="tickets",
    color="category"
)

fig_monthly_category.update_layout(
    height=520,
    xaxis_title="Month",
    yaxis_title="Tickets"
)

st.plotly_chart(
    fig_monthly_category,
    use_container_width=True,
    key="monthly_category"
)

st.divider()

st.subheader("Billing Routing Analysis")

helpdesk_start = pd.Timestamp("2025-09-14")

billing = tickets[
    (tickets["assigned_team"] == "Billing")
    & (tickets["created_at"] >= helpdesk_start)
].copy()

billing["resolved_team"] = billing.apply(
    lambda row: get_resolved_team(row, agents),
    axis=1
)

billing["misrouted"] = (
    billing["resolved_team"].notna()
    & (billing["resolved_team"] != "Billing")
)

billing_tickets = len(billing)
misrouted_tickets = int(billing["misrouted"].sum())
misrouting_rate = (
    misrouted_tickets / billing_tickets
    if billing_tickets
    else 0
)

transfer_count = int(
    billing["transfers"].fillna(0).sum()
)

transfer_cost = transfer_count * 305

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Billing Tickets",
        f"{billing_tickets:,}"
    )

with col2:
    st.metric(
        "Misrouted Tickets",
        f"{misrouted_tickets:,}"
    )

with col3:
    st.metric(
        "Misrouting Rate",
        f"{misrouting_rate:.1%}"
    )

with col4:
    st.metric(
        "Transfer Cost",
        f"₹{transfer_cost:,.0f}"
    )

billing_monthly = (
    billing.assign(
        month=billing["created_at"].dt.to_period("M").astype(str)
    )
    .groupby("month")
    .agg(
        tickets=("ticket_id", "count"),
        misrouted=("misrouted", "sum")
    )
    .reset_index()
)

billing_monthly["misrouting_rate"] = (
    billing_monthly["misrouted"]
    / billing_monthly["tickets"]
)

fig_billing = px.line(
    billing_monthly,
    x="month",
    y="misrouting_rate",
    markers=True
)

fig_billing.update_layout(
    height=400,
    xaxis_title="Month",
    yaxis_title="Misrouting Rate"
)

st.plotly_chart(
    fig_billing,
    use_container_width=True,
    key="billing_misrouting"
)

st.divider()

st.subheader("AI Ticket Categorizer")

model_data = tickets[
    tickets["text"].str.len() > 10
    & tickets["category"].notna()
].copy()

category_counts = model_data["category"].value_counts()

valid_categories = category_counts[
    category_counts >= 10
].index

model_data = model_data[
    model_data["category"].isin(valid_categories)
].copy()

model, validation_accuracy = train_model(
    model_data["text"],
    model_data["category"]
)

st.metric(
    "Agreement with Existing Categories",
    f"{validation_accuracy:.1%}"
)

st.caption(
    "This is holdout agreement with the existing intake categories, not human-verified accuracy."
)

message = st.text_area(
    "Enter a customer message",
    placeholder="Example: My payment was successful but my order has not arrived.",
    key="customer_message"
)

if message.strip():
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = probabilities.max()

    team_mapping = {
        "Billing & Payments": "Billing",
        "Delivery & Shipping": "Logistics",
        "Returns & Refunds": "Returns Desk",
        "Warranty & Repair": "Escalations & Warranty",
        "Product Enquiry": "Chat Frontline",
        "Connectivity": "Chat Frontline",
        "Charging & Battery": "Chat Frontline",
        "App & Firmware": "Chat Frontline",
        "Audio Quality": "Chat Frontline",
        "Account & Login": "Chat Frontline",
        "Other": "Frontline Review"
    }

    suggested_team = team_mapping.get(
        prediction,
        "Frontline Review"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Category",
            prediction
        )

    with col2:
        st.metric(
            "Suggested Team",
            suggested_team
        )

    with col3:
        st.metric(
            "Confidence",
            f"{confidence:.1%}"
        )

st.divider()

st.subheader("Human Validation Sample")

st.write(
    "Review the AI prediction for each sampled ticket. "
    "Select Correct or Incorrect after reading the customer message."
)

if "validation_sample" not in st.session_state:
    validation_source = tickets[
        tickets["customer_message"].str.len() > 20
    ].copy()

    validation_sample = validation_source.sample(
        n=min(30, len(validation_source)),
        random_state=42
    ).copy()

    validation_sample["ai_prediction"] = model.predict(
        validation_sample["text"]
    )

    validation_sample = validation_sample[
        [
            "ticket_id",
            "customer_message",
            "category",
            "ai_prediction"
        ]
    ].copy()

    validation_sample["review_status"] = "Not checked"

    st.session_state.validation_sample = validation_sample

validation_display = st.session_state.validation_sample.copy()

edited_validation = st.data_editor(
    validation_display,
    column_config={
        "ticket_id": st.column_config.TextColumn(
            "Ticket ID",
            disabled=True
        ),
        "customer_message": st.column_config.TextColumn(
            "Customer Message",
            disabled=True,
            width="large"
        ),
        "category": st.column_config.TextColumn(
            "Existing Category",
            disabled=True
        ),
        "ai_prediction": st.column_config.TextColumn(
            "AI Prediction",
            disabled=True
        ),
        "review_status": st.column_config.SelectboxColumn(
            "Human Review",
            options=[
                "Not checked",
                "Correct",
                "Incorrect"
            ],
            required=True
        )
    },
    hide_index=True,
    use_container_width=True,
    key="validation_editor"
)

st.session_state.validation_sample = edited_validation

checked = (
    edited_validation["review_status"] != "Not checked"
)

checked_count = int(checked.sum())

incorrect_count = int(
    (
        edited_validation["review_status"] == "Incorrect"
    ).sum()
)

human_error_rate = (
    incorrect_count / checked_count
    if checked_count
    else 0
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Sample Size",
        len(edited_validation)
    )

with col2:
    st.metric(
        "Checked Tickets",
        checked_count
    )

with col3:
    st.metric(
        "Human-Verified Error Rate",
        f"{human_error_rate:.1%}"
    )

if checked_count:
    st.write(
        f"{incorrect_count} of {checked_count} reviewed predictions "
        "were marked incorrect."
    )

st.divider()

st.subheader("Ticket Data")

display_columns = [
    "ticket_id",
    "created_at",
    "first_response_at",
    "resolved_at",
    "status",
    "channel",
    "customer_id",
    "order_id",
    "product_sku",
    "category",
    "priority",
    "assigned_team",
    "agent_id",
    "transfers",
    "csat_score",
    "refund_amount_inr",
    "refund_reason_code",
    "replacement_issued"
]

available_columns = [
    col for col in display_columns
    if col in tickets.columns
]

st.dataframe(
    tickets[available_columns].tail(100),
    use_container_width=True,
    height=500
)