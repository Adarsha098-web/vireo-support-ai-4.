# Vireo Support AI

An AI-assisted support analytics and ticket-routing prototype built for Vireo Audio.

## Objective

Analyze support ticket volume, identify routing problems, and prototype an AI-assisted ticket categorization workflow that can help route tickets to the appropriate support team.

## Business Problem

Vireo Audio receives support tickets across multiple channels and teams. The initial assignment team does not always match the team that ultimately resolves the ticket.

The project analyzes this routing behavior and evaluates whether improving categorization and routing could reduce avoidable transfers before additional headcount is added.

## Dataset

The analysis covers tickets from January 2025 through June 2026.

The project uses:

- tickets.csv
- agents.csv
- customers.csv
- orders.csv
- products.csv
- Support policy information

The primary analysis contains **11,641 tickets** within the January 2025 to June 2026 analysis period.

## Features

### Support Analytics

- Monthly ticket volume
- Category volume
- Team volume
- Monthly category trends
- Ticket-level data exploration

### Billing Routing Analysis

The prototype analyzes Billing tickets during the current helpdesk period beginning September 14, 2025.

It measures:

- Billing ticket volume
- Apparent misrouting
- Misrouting rate
- Internal transfers
- Estimated transfer cost

### AI Ticket Categorizer

The prototype uses **TF-IDF** text features and **Logistic Regression** to predict the support category from customer ticket text.

The system also provides:

- Predicted category
- Suggested support team
- Prediction confidence
- Human validation workflow

The model's holdout score measures agreement with existing intake categories. It is not presented as independently verified human accuracy because the source categories may contain labeling and routing issues.

## Business Finding

During the current helpdesk period:

- **1,558** tickets were initially assigned to Billing
- **482** were ultimately resolved by another team
- Apparent Billing misrouting was **30.9%**
- **615** internal transfers were recorded
- Transfer cost at the policy rate of ₹305 per transfer was approximately **₹187,575**

A modeled scenario reducing apparent Billing misrouting from **30.9% to 15%** represents approximately **248 fewer misrouted tickets** and an estimated **₹96,628 transfer-cost opportunity** over the analyzed period.

This is a modeled opportunity rather than guaranteed savings.

## Recommendation

Use AI-assisted categorization as a routing aid and address routing/process leakage before using raw queue volume as the sole basis for additional headcount.

High-confidence predictions can support routing, while low-confidence cases should remain subject to human review.

## Validation

The model achieved **83.4% holdout agreement** with the existing intake categories.

A separate manual review of **30 tickets** found:

- **30** tickets reviewed
- **11** incorrect predictions
- **36.7% human-reviewed error rate**

The validation sample is small and should not be treated as a production accuracy estimate.

## Running the Project

Create and activate a Python virtual environment, install the dependencies, and start Streamlit.

```bash
pip install -r requirements.txt
streamlit run app.py
