# Predictive Modelling & Agent-Based Simulation of Pneumonia Risk and Misinformation Impact

## Overview
This project integrates AI (machine learning methods such as logistic regression and XGBoost), NLP and agent-based modelling to explore pneumonia risk and detect misinformation to analyse its behavioural impact on care-seeking. The model simulates healthcare decision-making dynamics by incorporating misinformation exposure, social media behaviour and trust in medical systems.

## Project Components

### 1. Data Collection
* Chest X-ray Dataset from Kaggle
* Social Media Data via Twitter scraping (SNScrape or Tweepy) and Reddit
* Wikipedia
* HealthVer dataset
* News Articles via NewsAPI.org

### 2. Misinformation Detection Using NLP
* Uses TextBlob for sentiment and subjectivity scoring
* Flags potentially misleading or emotionally charged posts
* Prepares input for agent-based simulation (misinformation exposure score)

### 3. Pneumonia Detection from X-ray Images
* Image preprocessing with ImageDataGenerator (resize, normalise and augment)
* Flattened image data used with Logistic Regression and XGBoost
* Evaluation via accuracy and classification report

### 4. Agent-Based Simulation (ABM)
* Simulates patients and clinicians as agents
* Patient behaviour influenced by symptom severity, misinformation exposure, and trust in clinicians
* Clinicians diagnose using the trained ML model
* Measures changes in care-seeking behaviour over time

### 5. Misinformation Impact Analysis
* Quantifies how misinformation reduces symptom reporting and care-seeking
* Visualises behavioral trends over time and differences between exposed and unexposed groups

## Future Interventions
### Gift-Giving as Social Incentive and Engine of Social Contagion
To counter health misinformation and encourage timely care-seeking, an intervention that uses gift-giving as a catalyst for positive social contagion may be a promising avenue. Users who engage with or share verified health information, particularly content promoting pneumonia prevention, vaccination and symptom awareness may receive social and material rewards that include:

* Digital gift cards, recognition badges and exclusive content
* Public acknowledgment within online communities
* Invitations to expert Q&As and digital health events

These rewards are designed not just to incentivise individuals but to spark credibility cascades (chains of influence where seeing peers receive recognition or benefits for engaging with trustworthy content motivates others to do the same).

To ensure equity, the intervention may explicitly include support for disabled individuals and non-English speakers who often face systemic barriers to care and digital engagement. Content may be delivered in multiple languages, with translation models and culturally adapted phrasing that are optimised for assistive technologies (e.g., screen readers, simplified visual interfaces and alt text). Incentives may be designed with accessibility in mind, offering both digital and non-digital reward options to include users with limited internet access or varying physical and cognitive abilities.

AI models may help identify users well-positioned to trigger these cascades based on network structure, posting behavior and susceptibility to misinformation. They may also recommend personalised health content aligned with a user’s values, increasing the likelihood of uptake and onward sharing.

A gender-aware strategy may emphasise support for women as powerful nodes in these social networks. Women often play influential roles in how health information flows online and within a household. By targeting rewards and leadership opportunities to women who share verified content, the intervention would seek to amplify their impact, using gift-giving not only to motivate behaviour but to seed trust and accuracy into entire communities.

This approach treats information behavior as contagious and leverages trust, identity and social visibility to promote the viral spread of credible health knowledge across ideological and demographic boundaries.

### Key Metrics to Evaluate Gift-Giving & Social Contagion Effects
To assess how AI-enabled gift-giving promotes the spread of verified health information and influences care-seeking, metrics may include:

* Post-Gift Sharing Multiplier
   Average number of verified health shares or engagements triggered by a rewarded user’s action, measuring the strength of social contagion sparked by gift incentives.
* Cross-Cluster Spread
  Percentage of information cascades that successfully reach across ideological or demographic clusters, indicating how well the intervention bridges social divides and counters echo chambers.
* Misinformation Sharing/Sympathy Drop Rate
  Reduction in engagement with misinformation following exposure to gifted verified content, capturing the intervention’s effectiveness in dampening misinformation spread.
* Simulated Symptom Reporting Rate
  Percentage increase in accurate symptom reporting among agents after receiving or witnessing gift-based incentives, reflecting improved health awareness and honesty.
* Simulated Clinician Contact Rate
   Percentage increase in care-seeking behaviours such as contacting a virtual clinician among agents exposed to socially rewarded verified content, indicating real-world potential for improving health outcomes.

### Policy Impact
This work may provide actionable insights for public health agencies and policymakers that might include:

* Digital Health Campaign Design
   Offering a tested model for incorporating AI-powered simulations and misinformation detection into national health communication strategies.
* Targeted Incentive Programmes:
  Demonstrate how algorithmically personalised social rewards can encourage accurate health messaging particularly across ideologically diverse and gendered communities.
* Health Equity Policy
  Highlight the importance of engaging women as trusted health communicators who inform gender-responsive public health planning.
* Social Media Regulation
  Provide a framework for collaboration between health authorities and digital platforms to flag, de-rank and incentivise correction of misinformation in ways that do not alienate users.

## Setup Instructions
```plaintext
git clone https://github.com/yourusername/pneumonia-misinformation-model.git
cd pneumonia-misinformation-model
pip install -r requirements.txt
```
## Make sure to:
Set up API keys for Twitter and NewsAPI (if applicable)
Download and extract the chest X-ray dataset into the data/ folder

## Project Structure
```plaintext
├── data/
│   ├── chest_xray/
├── src/
│   ├── preprocess_images.py
│   ├── train_models.py
│   ├── nlp_misinformation.py
│   ├── agent_based_simulation.py
├── notebooks/
├── visualisations/
├── README.md
├── requirements.txt
```
## Sample Results
Logistic Regression Accuracy: 78%
XGBoost Accuracy: 82%
Misinformation Impact: Up to 11% drop in care-seeking behaviour in high-exposure scenarios

## Dependencies
* scikit-learn
* xgboost
* tensorflow\keras
* matplotlib, seaborn
* textblob
* mesa (for ABM)
* snscrape or tweepy

## Contact
For questions or collaboration requests, please contact me here or open an issue.
# Pneumonia
