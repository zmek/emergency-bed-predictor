---
title: 'Introducing the modelling problem'
teaching: 10
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is 'emergency demand for beds'?
- Why is modelling emergency demand for beds important?
- What is a predictive model? 

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain what is meant by 'emergency demand for beds'?
- Introduce why this is important for hospitals 
- Explain what a predictive model is 
- Evaluate what a model needs to produce to be useful 

::::::::::::::::::::::::::::::::::::::::::::::::

## What is 'emergency demand for beds'?

There are two ways people are admitted to hospital: 
- planned (or elective) admissions 
- emergency admissions, typically through the Accident & Emergency Department, which is referred to by hospitals as the ED (Emergency Department). Some patients are admitted after visiting Same Day Emergency Care (SDEC). 

Hospitals can usually plan well for electives. Emergencies are, by definition, unplanned. Most people are discharged after visiting the ED or SDEC, but about 10% are admitted to an emergency bed. Their bed requirement is what we mean when we say 'emergency demand for beds'

## Why is modelling emergency demand for beds important?

Hospitals find it useful to predict how many beds they need on a given day because:
- they must make sure they do need admit more people than they can safely care for. If they think they can no longer provide safe care, they take measures like diverting ambulances to other places or cancelling planned procedures
- this planning has to take account of emergency patients who have not arrived yet as well as those currently in the ED or SDEC who will need admission
- knowing levels of demand helps plan for best use of resources like staffing
- if they know which specialties are in demand, they can take steps to discharge more patients from those areas

The demand for emergency beds varies 
- by time of day, day of week and season
- in terms of the acuity of the patients - how sick they are
- in terms of which specialty they need to be admitted to - medical, surgical, paediatric etc

Hospitals typically deal with this using simple heuristics eg taking yesterday's emergency bed numbers and assuming today they will admit the same number.

However, now that hospitals have electronic patient records, they could do more with real-time data. Typically, bed planners only know about new demand when someone in the ED or SDEC creates a bed request for an incoming patients. By that time, the patient may have been in the hospital for hours. Real-time data means that bed planners could be notified much earlier. 

## What is a predictive model? 

A predictive model is a statistical tool that analyses historical data to forecast future outcomes.

A predictive model can help  
- give some knowledge of the likely number of beds needed today, but also, because it is based in statistics:
- provide a sense of uncertainty around that number

A point estimate gives just one possible number (of beds). This doesn't show how likely that outcome is compared to others. A probability distribution shows all possible outcomes and how likely each one is.

Real-time data is not necessary for predictive models. They can be trained on historical datasets, and then used to make predictions about the numbers of patients to expect today. However, if a hospital is able to feed real-time data about patients into a model, it will give a better sense of today's demand, rather than a statistically probable day


## What does a model need to produce to be useful to bed managers? 

It is quite common for analysts using statistics or Machine Learning to produce models that predict each patient's probability of admission. However, this is not useful for bed managers because, when managing capacity, they are less interested in whether one or another patient will be admitted, and more interested in overall numbers. 

However, individual-level predictions are very useful especially if each patient's probability can be based on their real-time data, so we don't want to reject the individual level data. 

Instead we will show how to aggregate from individual-level data to overall bed numbers. 

*The requirements for a good model for bed planners*

- **Produces output at aggregate level rather than individual:**
  The output variable is a number of beds 

- **Takes account of patients who are yet to arrive:**
  A picture comprising only patients currently in the hospital is incomplete. When thinking in the morning about demand over the whole day, they want to take account of later arrivals

- **Has a breakdown of demand by specialty and sex:**
  Predictions of demand across the whole hospital is somewhat useful, but it is not actionable. Only once you know how many specific types of beds are needed can you take action. For example, they might decide to reconfigure a bay (a set of four beds with allocated staff) on a ward from male to female. 

- **Excludes patients they already know about:**
  Bed planners already have a list of patients with bed requests. Predictive models are most useful if they can show coming demand that bed planners don't already know about.




::::::::::::::::::::::::::::::::::::: keypoints 

- Predictive models can help bed planners plan how many emergency beds they will need 
- This is particularly useful when it can show incoming demand from patients they do not know about yet
- Real-time data from electronic health records can inform the predictions
- Overall bed counts are more useful to bed planners than predictions for individual patients

::::::::::::::::::::::::::::::::::::::::::::::::

