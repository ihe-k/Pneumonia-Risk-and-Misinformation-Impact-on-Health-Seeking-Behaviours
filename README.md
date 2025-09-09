# Predictive Modelling & Agent-Based Simulation of Pneumonia Risk and Misinformation Impact

## Overview
This project integrates AI (machine learning methods such as logistic regression and XGBoost), NLP and agent-based modelling to explore pneumonia risk and detect misinformation to analyse its behavioural impact on care-seeking. The model simulates healthcare decision-making dynamics by incorporating misinformation exposure, social media behaviour and trust in medical systems.

[Link to App](https://pneumonia-risk-and-misinformation-impact-on-health-seeking-beh.streamlit.app/)

## Project Components

### 1. Data Collection
* Chest X-ray Dataset from Kaggle
* Social Media Data via Reddit
* Wikipedia
* HealthVer dataset

### 2. Pneumonia Detection from X-ray Images: Deep Learning and Machine Learning Classification
Detecting pneumonia from medical images consists of the following process:

* Image preprocessing with ImageDataGenerator (resize, normalise and augment)
* Flattened image data used with Logistic Regression and XGBoost
* Evaluation via accuracy and classification report

#### Image Preprocessing
Images are resized to a consistent size and pixel intensities are normalised as neural networks require input data that are a consistent shape and scale.

#### Machine Learning Classifiers (Logistic Regression and XGBoost)

Logistic regression is a linear classifier that outputs a probability of pneumonia using the sigmoid function:

<img src="https://github.com/ihe-k/Pneumonia-Risk-and-Misinformation-Impact-on-Health-Seeking-Behaviours/blob/feature/clean-env-setup/P2_Eq_1.png?raw=true" width="160" />
Where:

* *z* = *w*<sub>1</sub>*x*<sub>1</sub> + *w*<sub>2</sub>*x*<sub>2</sub> + ... + *w*<sub>n</sub>*x*<sub>n</sub> + *b* -> linear combination of inputs
* *p* is the probability that the chest X-ray (i.e. input) shows pneumonia

The model consequently returns a value between 0 and 1 which after applying a threshold of 0.5, allows a prediction of either pneumonia or no pneumonia.

Extreme Gradient Boosting, XGBoost, is an ensemble of decision trees that works by minimising a loss function.  Each tree corrects the errors of the previous one and are optimised for speed and accuracy.  XGBoost sums the raw scores of all trees before passing a prediction through the sigmoid function which is converted to a probability.  After applying a threshold of 0.5, the output produced (a value between 0 and 1), allows a prediction of either pneumonia or no pneumonia.

#### Model Evaluation
Models are evaluated using an accuracy and classification report that includes:

* Precision: The number of correct predicted positives
* Recall: The number of actual positives that were caught
* F1 Score: A balance of precision and recall

### 3. Misinformation Detection Using NLP
Sentiment analysis and NLP tools are implemented to analyse social media posts (e.g., Reddit comments) for misinformation regarding pneumonia:

* TextBlob is utilised for sentiment and subjectivity scoring
* Potentially misleading or emotionally charged posts are flagged
* Input for agent-based simulation (misinformation exposure score) is prepared

TextBlob is a Python NLP library that provides sentiment analysis, subjectivity and tokenisation. Sentiment analysis highlights negative sentiment (posts illustrating distrusts in clinicians, conspiracy theories or panic posts); highly subjective text (e.g., opinion-based content) and misleading posts that are overly negative.

The NLP model outputs flags for misinformation as well as misinformation scores between 0 and 1, which are passed into the Agent-Based Model (ABM) as the patient's initial misinformation exposure.  

### 4. Misinformation Model: ABM
ABM simulates individual agents (patients and clinicians) who interact over time in a spatial environment, where misinformation might spread and impact behaviour. Patients are the main decision-makers in this model and seek care based on multiple attributes:

* Symptom Severity: Illustrates how sick the patient feels and is randomised as well as modified over time.
* Trust in Clinicican: Highlights the trust a patient invests in medical advice (it is dynamic and may increase with clinician interaction).
* Misinformation Exposure: The extent of a patient's exposure to false health information (e.g., 'you do not need to see a doctor for a cough')
* Care-Seeking Behaviour: The likelihood a patient will seek care (updated dynamically).

Clinicians interact with patients to increase trust and reduce misinformation.  Patient behaviour evolves based on rules as well as randomness.  The goal of this model is to understand the ways that misinformation affects care-seeking behaviour especially under various conditions like symptom severity, exposure to health misinformation, location (urband and rural) as well as trust in clinicians.  Each time step, a patient updates their internal state:

* Misinformation decreases care-seeking
* Trust and high symptom severity increases care-seeking

After N steps (i.e. 30), care-seeking in response to misinfomation and trust in clinician is explored).  Clinicians diagnose using the trained ML model.

Adjusting the 'number of patient agents' affects population size and the realism of interactions; adjusting the number of clinicians determines the number of patients that are treated with increased trust or corrected misinformation and adjusting the misinformation exposure level allows an investigation into the impact of fifferent misinformation levels on care-seeking behaviours.

#### Simulation Modes
* Stepped: Collects data at each step (e.g., daily or weekly) to analyse how behaviour evolves over times
* Non-Stepped: Only reports snapshot of the population state at the final step after full simulation

### 5. Misinformation Impact Analysis
Quantifies how misinformation reduces symptom reporting and care-seeking (R² and p-values give statistical validity to observed relationships):

- Relationship between Symptom Severity and Care Seeking Behaviour (left plot): The colour gradient of the points represents misinformation exposure. 
- Relationship between Misinformation Exposure and Care Seeking Behaviour (middle plot):  The colour gradient of the points represents patient trust in a clinician.
- Relationship between Trust in Clinician and Care Seeking Behaviour (right plot). The colour gradient of the dots also represent Misinformation Exposure.

These graphs help identify how misinformation and trust in clinicians might affect a patient's behaviour. For example:

* Trust in Clinician vs Care-Seeking: Patients with low trust in clinicians and high misinformation exposure might be less likely to seek care.  This would be visible as a higher proportion of dark to light green dots in the lower-left region of the graph.
* Higher symptom severity combined with lower misinformation exposure and higher clinician trust appears to correlate with increased care-seeking behaviour (dark purple dots towards the top-right).
* The analyses suggest that misinformation exposure significantly reduces the likelihood of seeking care, even when symptoms are severe. This highlights the importance of public health interventions aimed at combating misinformation to improve care access and health outcomes.  Future investigation may include identification of interventions that target variables with strong relationships (e.g., to improve care-seeking behaviour, health care organisations increase corrective efforts to counteract misinformation exposure levels which significantly impacts care-seeking hehaviour).

In the simulation script, the following components are crucial for the graphs:

* Agent Creation and Data Collection:
- In the Misinformation Model, agents are created with attributes like Symptom Severity, Care Seeking Behaviour, Trust in Clinician, and Misinformation Exposure.
- These agent attributes are then collected over time using the DataCollector.  The DataCollector tracks the changes in these variables across simulation steps, which are later used for the visualisations.

In the script, once the simulation is triggered, the model runs for 30 steps.  Each step represents an agent-based model simulation run where each agent's behaviour is updated based on their attributes and interactions. After each step, the model collects data using the datacollector.

### Use Cases for the Simulation

* Public Health Policy Testing: Exploration of the way clinician capacity or misinormation campaigns affect care-seeking outcomes
* Educating Policymakers: Illustrative examples that highlight the reasons fighting misinformation or buillding clinician trust is essential
* Modelling Human Psychology: Traditional compartmental disease models do not capture beliefs, trust and behavours change in the way this ABM is able to explore psychosial and behavioural aspects that are critical in modern healthc rises.

## Future Interventions
### Gift-Giving as Social Incentive and Engine of Social Contagion
To counter health misinformation and encourage timely care-seeking, an intervention that uses gift-giving as a catalyst for positive social contagion may be a promising avenue. Users who engage with or share verified health information, particularly content promoting pneumonia prevention, vaccination and symptom awareness may receive social and material rewards that include:

* Digital gift cards, recognition badges and exclusive content
* Public acknowledgment within online communities
* Invitations to expert Q&As and digital health events

These rewards are designed not just to incentivise individuals but to spark credibility cascades (chains of influence where seeing peers receive recognition or benefits for engaging with trustworthy content motivates others to do the same).

To ensure equity, the intervention may explicitly include support for disabled individuals and non-English speakers who often face systemic barriers to care and digital engagement. Content may be delivered in multiple languages, with translation models and culturally adapted phrasing that are optimised for assistive technologies (e.g., screen readers, simplified visual interfaces and alt text). Incentives may be designed with accessibility in mind, offering both digital and non-digital reward options to include users with limited internet access or varying physical and cognitive abilities.

AI models may help identify users well-positioned to trigger these cascades based on network structure, posting behaviour and susceptibility to misinformation. They may also recommend personalised health content aligned with a user’s values, increasing the likelihood of uptake and onward sharing.

A gender-aware strategy may emphasise support for women as powerful nodes in these social networks. Women often play influential roles in how health information flows online and within a household. By targeting rewards and leadership opportunities to women who share verified content, the intervention would seek to amplify their impact, using gift-giving not only to motivate behaviour but to seed trust and accuracy into entire communities.

This approach treats information behaviour as contagious and leverages trust, identity and social visibility to promote the viral spread of credible health knowledge across ideological and demographic boundaries.

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
├── chest_xray/
├── Graphs/
├── train_pneumonia.py
├── train_pneumonia_model.py
├── Saved_Train_Model/
│   ├── pneumonia_log_reg.pkl
│   ├── pneumonia_xgb.pkl
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
