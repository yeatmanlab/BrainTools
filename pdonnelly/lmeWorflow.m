% LME Model Workflow
% Patrick Donnelly, University of Washington, Jan 16, 2017
%
% Used to systematically discover the best fitting model by determining
% the relationship and optimization of random/fixed effect variables.

%% Using data structure of lmeLong output - zoning in on test of interest
% refer to lmeLong for order of tests
test = stats(3);

%% Gather simple linear fit
% Fixed: time
% Random: subject grouped by intercept
% in output of lme model, a random effect has significant effect if the confidence
% interval does not include zero, implying that its influence is nonzero and 
% not overparameterized
simple_linear = fitlme(test.data_table, 'score ~ long_var + (1|sid)')
% Check residuals in plot
plotResiduals(simple_linear, 'fitted');

%% Check alternate linear model
% Fixed: time
% Random: subject, time-grouped by subject(independent)
% pay attention to pValues of fixed effects coeff & confidence interval of random effects
altlme = fitlme(test.data_table, 'score ~ 1 + long_var +(1|sid) + (long_var - 1|sid)')

%% Formally compare the models using the compare() function
% AIC and BIC values will determine the superior model
% the lower the AIC and BIC values, the better the model
% BIC values are consider more important
compare(simple_linear, altlme, 'CheckNesting', true) 

%% Alternate linear model with add'l random effect
% Fixed: time
% Random: time, subject each grouped by intercept
% Check for intercept of random terms - for overparamterization 
% overparameterization might reveal correlation
linear2 = fitlme(test.data_table, 'score ~ 1 + long_var + (1|sid) + (1|long_var)')

%% Compare new model to original basic linear
compare(simple_linear, linear2, 'CheckNesting', true)

%% New linear model 
% Checking for correlation
% Fixed: time
% Random: time, grouped by subject
% Check for correlation in output
lme_linear3 = fitlme(test.data_table, 'score ~ 1 + long_var + (1 + long_var|sid)')

%% Compare correlated linear with basic linear
compare(simple_linear, lme_linear3, 'CheckNesting', true)

%% Compare previous outcome with previous most significant model
compare(altlme, lme_linear3, 'CheckNesting', true)

%% Try a quadratic term with most successful linear model
quad = fitlme(test.data_table, 'score ~ 1 + long_var^2 + (1 | sid) + (long_var-1| sid)')
% quad = fitlme(test.data_table, 'score ~ 1 + long_var^2 + (1|sid)')

% Compare with linear
compare(altlme, quad, 'CheckNesting', true)

%% Try a cubic term with the most successful linear model
% cube = fitlme(test.data_table, 'score ~ 1 + long_var^3 + (1 | sid) + (long_var-1| sid)')
cube = fitlme(test.data_table, 'score ~ 1 + long_var^2 + long_var^3 + (1|sid)')
% Compare with linear
compare(simple_linear, cube, 'CheckNesting', true)

