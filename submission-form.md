# Vireo Audio — Submission Form

## What did you build, and what business outcome does it move?

I built a Streamlit dashboard with an ML-assisted ticket categorizer using TF-IDF and Logistic Regression. It displays monthly ticket volumes, category and team breakdowns, Billing routing analysis, predicted categories, suggested teams and confidence scores.

In the analyzed current-helpdesk period, 482 of 1,558 tickets initially assigned to Billing were resolved by another team (30.9%). Those tickets recorded 615 transfers, representing ₹187,575 at ₹305 per transfer.

The business goal is to reduce apparent Billing misrouting from 30.9% to 15%. At the observed volume and transfer rate, this represents approximately 248 fewer misrouted tickets and a modeled ₹96,628 transfer-cost opportunity over the same period. These are estimates, not guaranteed savings.

## What does one run cost, and what would a month cost?

The classifier runs locally without paid model or API calls.

- Paid inference cost per ticket: ₹0
- Monthly volume: 650 × 52 ÷ 12 ≈ 2,817 tickets
- Estimated monthly paid inference cost: 2,817 × ₹0 = ₹0

Hosting, infrastructure, maintenance and human-review costs are excluded.

## How do you know it works?

The model achieved 83.4% holdout agreement with the existing intake categories. Because these labels may be inaccurate, I also used a separate manual-review workflow.

I manually reviewed a 30-ticket sample and marked 11 predictions incorrect, giving a 36.7% human-reviewed error rate. The sample is small and is not a reliable estimate of production accuracy. Errors included confusion between closely related categories such as returns/refunds, warranty/repair, connectivity and billing/payment issues.

## Did you change, narrow or push back on the client's ask?

I retained the requested categorization and monthly breakdowns but expanded the analysis to investigate routing problems. Raw first-assigned ticket volume alone may overstate a team's actual workload. I focused the business case on Billing routing and transfer costs rather than automatically assigning two hires to the team with the largest queue.

## What is wrong with what you are handing us?

The model was trained on potentially unreliable intake categories. The manual validation sample is small. Suggested teams are based on category-to-team rules, not a production routing system. A different resolving team is only a proxy for initial misrouting, and some transfers may have been necessary. Historical data also contains a helpdesk migration, so transfer analysis is restricted to the current-helpdesk period.

## What did you deliberately leave out, and why?

I did not build live helpdesk integration, automatic ticket assignment or a production deployment. I prioritized a working local prototype, interpretable business analysis and validation within the available time.

## Anything you built or found that nobody asked for?

I added a Billing routing analysis, transfer-cost estimate, prediction confidence display and manual-validation workflow. These help connect categorization quality to an operational outcome.

## What did you use AI for?

I used ChatGPT to help plan the analysis, develop and debug the Python/Streamlit prototype, interpret results and draft documentation. The application itself uses a local scikit-learn TF-IDF and Logistic Regression model rather than a paid generative-AI API.

I revised the app after testing, corrected the development environment and added human validation to distinguish model agreement with existing labels from manually reviewed performance.

**Screen recording:** [Add Google Drive link]

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. Run `pip install -r requirements.txt`, followed by `streamlit run app.py`. The application reads `data/tickets.csv` and `data/agents.csv`.
2. The 83.4% model score measures agreement with existing labels, not verified accuracy. The separate 30-ticket review recorded an error rate of 36.7%.
3. The ₹96,628 opportunity is a modeled scenario based on historical transfer costs. It must be tested in a controlled pilot before being treated as realized savings.

## Honest hours spent

6 hours

## GitHub Repo Link

[Add public GitHub repository URL]