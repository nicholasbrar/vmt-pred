# Household Vehicle Miles Traveled (VMT) Prediction

The current model for predicting household vehicle miles traveled (VMT) faces many limitations. While exclusively using household features, the regression model achieves a low weighted R² score of around .05. When applying a log-transform to stabilize the skewed nature of the VMT (some households hardly travel while others travel tens of thousands of miles/year), the model achieves a weighted R² score of around .52. This approach seems much more stable, but prevents the accurate recovery of the original scale in miles traveled, because the other dependent variables are also transformed. This is promising, but operating in log-space makes the model far from useful in the real-world. Future work will involve incorporating vehicle-specific features (vehicle type, vehicle miles), and seeing if these features capture the essence of the data better. Additionally, better feature engineering and exploring different modeling approaches might give better performance.

---

## Updates

**Update:** Added winsorization to some key features to reduce the impact of severe outliers on the model. The model scores very slightly better with an R² score of around .06.

**Update 2:** Adding vehicle and person level data further improved the outcome. Sitting at an R² score of around .08, which is still modest but showing some incremental improvement.
