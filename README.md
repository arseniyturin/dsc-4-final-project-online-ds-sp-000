
# Predicting Patient Condition Based on Drug Review


## Introduction

In this lesson, we'll review all of the guidelines and specifications for the final project for Module 4. 

## Objectives
You will be able to:
* Describe all required aspects of the final project for Module 4
* Describe all required deliverables
* Describe what constitutes a successful project
* Describe what the experience of the project review should be like

## Final Project Summary

You've made it all the way through one of the toughest modules of this course, and demonstrated a solid understanding of the principles of Deep Learning. You must have an amazing brain in your head!

<img src='https://raw.githubusercontent.com/learn-co-curriculum/dsc-4-final-project/master/brain.gif' height=40% width=40%>

For this module's final project, you'll put everything you've learned together to build a Deep Neural Network that trains on a large dataset for classification on a non-trivial task! This project will include:

* Selecting a problem 
* Sourcing an appropriate dataset
* Setting up your project (directory structure, etc)
* Building, training, tuning, and evaluating a Deep Neural Network
* Explaining your methodology and findings in a clear, concise manner

Let's get started by examining the dataset requirements for this project.

## The Dataset

For this module's project, the dataset will be heavily tied to the problem you are trying to solve. We recommend that you base your project around one of the three following subdomains in Deep Learning which you now have experience with:

* Traditional  analytics (classification or regression tasks)
* Computer Vision
* Text Classification/NLP

### Picking a Reasonable Problem

Note that in respect to this project, all datasets and problems are not created equal--while you could likely build a working model for just about any dataset you find in theory, in practice, you'll find that many datasets have dimensionality issues that make them intractable for training without spending hundreds or even thousands of dollars training your model on a professional server cluster filled with high-end GPUs.

A good litmus test for checking a project's feasibility is to head over to Kaggle or do a quick Google search to see if anyone else has already solved this problem. If they have, then it's likely that you can, too! Just remember, you only have access to a local machine for this project, not a server cluster, so the problem should be one that can be solved on your own laptop!

Here are some caveats you should consider when selecting your dataset:

#### A Note on Computer Vision Datasets

**Try to stay away from color images, or images that are larger than 40x40 pixels**. Convolutional Layers are very expensive, and most models can still make successful classifications on grainy, black-and-white images just fine. Pictures that are too large add a bunch of needless dimensionality to the model--remember, every single pixel in the model is a dimension! Similarly, since color images are Rank-3 Tensors (3-dimensional arrays contain Red, Green, and Blue values for each pixel), they also needlessly triple dimensionality without adding important information to your model in most cases.

#### Aim for a Proof of Concept

With Deep Learning, data is king--the more of it, the better. However, the goal of this project isn't to build the best model possible--it's to demonstrate your understanding by building a model that works. The true goal of this project is to gain experience with Deep Learning and to build a portfolio project you can be proud of, and that doesn't necessarily require a model with incredibly high accuracy. You should try to avoid datasets and model architectures that won't run in reasonable time on your own machine. For many problems, this means downsampling your dataset and only training on a portion of it. Once you're absolutely sure that you've found the best possible architecture and other hyperparameters for your model, then consider training your model on your entire dataset overnight (or, as larger portion of the dataset that will still run in a feasible amount of time). 

At the end of the day, we want to see your thought process as you iterate and improve on a model. A Project that achieves a lower level of accuracy but has clearly iterated on the model and the problem until it found the best possible approach is more impressive than a model with high accuracy that did not iteration. We're not just interested in seeing you finish a model--we want to see that you understand them, and can use this knowledge to try and make them better and better!


#### Preexisting Datasets

As you start exploring datasets that are appropriate for Deep Learning, you'll probably start to see some of the same datasets mentioned again and again, such as CIFAR10. For this project, it is acceptable to use popular preexisting datasets. **It is also acceptable to use datasets that you've found on popular websites such as Kaggle--you'll find a very active Deep Learning community on that website and plenty of awesome datasets that are perfect for this sort of project!**

#### Sourcing Your Own Dataset

If you so choose, you are also welcome to source your own dataset for this project, although we strongly advise you to think carefully about whether this is worth the time before attempting this! You'll likely need thousands of examples, and scraping google images or other websites can sometimes be more trouble than it's worth. If you feel up to the task, then you are more than welcome to source your own dataset through scraping. However, we strongly encourage you to search the web for preexisting datasets that would work for your chosen task before attempting to source your own, as they likely already exist, and will save you a ton of time debugging your scraping code or getting an API to work.  **If you plan on sourcing your own dataset for this project, please clear this with your instuctor first!**
 
 
 #### Avoid Generative Models
 
After the end of the Deep Learning module, you may be tempted to try building a Generative Model such as a Generative Adversarial Network, Variation Autoencoder, or Sequence Generation model. Although you theoretically know enough to attempt such problems, in practice, these models are much too computationally intensive for you to see any meaningful results on a local machine in the time allotted. For reference, most GANs for image generation need to train for a minimum of 3 days straight on a server cluster with 64 high-end GPUs before showing any meaningful results! The other issue with generative models is that they are unsupervised, so it is impossible to generate any sort of accuracy or performance metrics. **For this reason, you must stick to supervised learning and only build discriminative models for this project. No generative models will be approved.**

## The Deliverables

There will be four deliverables for this project:

1. A well documented **Jupyter Notebook** containing any code you've written for this project and comments explaining it. This work will need to be pushed to your GitHub repository in order to submit your project.
2. A short **Keynote/PowerPoint/Google Slide presentation** (delivered as a PDF export) giving a high-level overview of your methodology and recommendations for non-technical stakeholders. Make sure to also add and commit this pdf of your non-technical presentation to your repository with a file name of presentation.pdf. 
3. A **blog post** (800-1500 words) about one element of the project - it could be the EDA, the feature selection, the choice of visualizations or anything else technical relating to the project. It should be targeted at your peers - aspiring data scientists.
4. A **Video Walkthrough** of your non-technical presentation. Some common video recording tools used are Zoom, Quicktime, and Nimbus. After you record your presentation, publish it on a service like YouTube or Google Drive, you will need a link to the video to submit your project.

## The Process

### 1. Getting Started

Please start by reviewing this document. If you have any questions, please ask them in slack ASAP so (a) we can answer the questions and (b) so we can update this repository to make it clearer.

Once you're done with the rest of the module, please start on the project. Do that by forking this repository, cloning it locally, and working in the student.ipynb file. Make sure to also add and commit a pdf of your presentation to the repository with a file name of `presentation.pdf`.

### 2. The Project Review

#### What to expect from the Project Review

Project reviews are focused on preparing you for technical interviews. Treat project reviews as if they were technical interviews, in both attitude and technical presentation *(sometimes technical interviews will feel arbitrary or unfair - if you want to get the job, commenting on that is seldom a good choice)*.

The project review is comprised of a 45 minute 1:1 session with one of the instructors. During your project review, be prepared to:

#### 1. Deliver your PDF presentation to a non-technical stakeholder. 
In this phase of the review (~10 mins) your instructor will play the part of a non-technical stakeholder that you are presenting your findings to. The presentation should not exceed 5 minutes, giving the "stakeholder" 5 minutes to ask questions.

In the first half of the presentation (2-3 mins), you should summarize your methodology in a way that will be comprehensible to someone with no background in data science and that will increase their confidence in you and your findings. In the second half (the remaining 2-3 mins) you should summarize your findings and be ready to answer a couple of non-technical questions from the audience. The questions might relate to technical topics (sampling bias, confidence, etc) but will be asked in a non-technical way and need to be answered in a way that does not assume a background in statistics or machine learning. You can assume a smart, business stakeholder, with a non-quantitative college degree.

#### 2. Go through the Jupyter Notebook, answering questions about how you made certain decisions. Be ready to explain things like:
    * "how did you pick the question(s) that you did?"
    * "why are these questions important from a business perspective?"
    * "how did you decide on the data cleaning options you performed?"
    * "why did you choose a given method or library?"
    * "why did you select those visualizations and what did you learn from each of them?"
    * "why did you pick those features as predictors?"
    * "how would you interpret the results?"
    * "how confident are you in the predictive quality of the results?"
    * "what are some of the things that could cause the results to be wrong?"

Think of the second phase of the review (~30 mins) as a technical boss reviewing your work and asking questions about it before green-lighting you to present to the business team. You should practice using the appropriate technical vocabulary to explain yourself. Don't be surprised if the instructor jumps around or sometimes cuts you off - there is a lot of ground to cover, so that may happen.

If any requirements are missing or if significant gaps in understanding are uncovered, be prepared to do one or all of the following:
* Perform additional data cleanup, visualization, feature selection, modeling and/or model validation
* Submit an improved version
* Meet again for another Project Review

What won't happen:
* You won't be yelled at, belittled, or scolded
* You won't be put on the spot without support
* There's nothing you can do to instantly fail or blow it

## Requirements

This section outlines the rubric we'll use to evaluate your project.

### 1. Technical Report Must-Haves

Your jupyter notebook should include all code written for this project. This includes any code for sourcing, cleaning, and preprocessing data. Your technical report should also contain a record of the various different hyperparameters you tried during the tuning process, and the results each achieved. Any data scientist given your technical report should be able to reproduce every step you took during the project from start to finish and achieve the same results, so don't forget to set a random seed for reproducibility!

As always, your jupyter notebook should be well-organized and easy to read, with clean, well-commented code as necessary.


### 2. Non-Technical Presentation Must-Haves

Just as with the other projects, you should also complete a 5-10 slide PowerPoint or Google Slides presentation that explains your problem, methodology, and results to non-technical stakeholders. This can be especially hard with Deep Learning--try not to get bogged down with technical jargon! Your slide deck should take ~5 minutes to go through and should contain graphics and avoid long blocks of text or code when possible. 

**_HINT_**: Keras provides [excellent documentation](https://keras.io/visualization/) on how to create a visualization of your neural network's architecture!

### 3. Blog Post

Please also write a blog post about your experience working on this project. This blog post should provide insight into the problem you are trying to solve and your dataset, any preprocessing steps required, and your approach to building and iteratively tuning your model. It should also contain an explanation of any problems, obstacles, or surprises you encountered during this project. The blog post should be between 800-1500 words and should be targeted at your peers - aspiring data scientists.

## Submitting your Project

You’re almost done! In order to submit your project for review, include the following links to your work in the corresponding fields on the right-hand side of Learn.

1. **GitHub Repo:** Now that you’ve completed your project in Jupyter Notebooks, push your work to GitHub and paste that link to the right. (If you need help doing so, review the resources [here](https://docs.google.com/spreadsheets/d/1CNGDhjcQZDRx2sWByd2v-mgUOjy13Cd_hQYVXPuzEDE/edit#gid=0).)
_Reminder: Make sure to also add and commit a pdf of your non-technical presentation to the repository with a file name of presentation.pdf._
2. **Blog Post:** Include a link to your blog post.
3. **Record Walkthrough:** Include a link to your video walkthrough.

Hit "I'm done" to wrap it up. You will receive an email in order to schedule your review with your instructor.

## Summary

The end of module projects and project reviews are a critical part of the program. They give you a chance to both bring together all the skills you've learned into realistic projects and to practice key "business judgement" and communication skills that you otherwise might not get as much practice with.

The projects are serious and important. They are not graded, but they can be passed and they can be failed. Take the project seriously, put the time in, ask for help from your peers or instructors early and often if you need it, and treat the review as a job interview and you'll do great. We're rooting for you to succeed and we're only going to ask you to take a review again if we believe that you need to. We'll also provide open and honest feedback so you can improve as quickly and efficiently as possible.

We don't expect you to remember all of the terms or to get all of the answers right. If in doubt, be honest. If you don't know something, say so. If you can't remember it, just say so. It's very unusual for someone to complete a project review without being asked a question they're unsure of, we know you might be nervous which may affect your performance. Just be as honest, precise and focused as you can be, and you'll do great!


