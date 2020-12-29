# 2D-linear-regression

In this project we will implement linear regression with two variables to fit some randomly generated data to a plan. You will find and a folder names 'code' the py version and the jupyter notebook version of this project for those who are familiar with jupyter.

We will use gradient descent to solve this problem.


# Let's import some libraries and initialize variables

    import numpy as np
    import matplotlib.pyplot as plt

    #Variables
    m=100
    
    
    #The first feature
    x1 = np.linspace(0, 10, m).reshape((m, 1)) 
    
    #The second feature
    x2 = np.linspace(0, 30, m).reshape((m, 1))+np.random.randn(m, 1) 
    
    #The output : y = input+noise, the noise is small because we want to fit data to a plan
    y = (x1+x2 +np.random.randn(m, 1)) 
    
    #first column is all ones by convention
    X = np.hstack((np.ones(x1.shape),x1,x2 )) 
    
    # theta parameter contains three variables
    theta = np.random.rand(3,1) 
    itterations = 100
    alpha = 0.001
    J = np.zeros((itterations))
  
J will store the cost function value at each itterations.



# Defining functions for computations

    #Hypothesis
     def h(theta):
         return X.dot(theta)

    #Cost function
    def computeJ(theta):
        return 1/(2*m)*np.sum(  (np.square( h(theta)-y) ) ) 

    #Gradient
    def gradient(theta):
        return (1/m)*X.T.dot(h(theta)-y)

    #Gradient descent
    def gradientDescent(alpha,itterations,theta):
        for i in range(0,itterations):
            J[i] = computeJ(theta)
            theta = theta - alpha*gradient(theta)
        return theta
        
  # Solution
    thetaSolution = gradientDescent(alpha,itterations,theta)
    Jsol = computeJ(thetaSolution)
    print('\u03B80=', thetaSolution[0], ',\u03B81=',thetaSolution[1],'\u03B82=', thetaSolution[2],'\nJ(\u03B8)=',Jsol)
    
θ0= [0.29486968] ,θ1= [0.14619083] θ2= [1.26388543]  <br/>
J(θ)= 0.47101937667523786
    
# Plot of the cost function according to number of itterations

    plt.plot(J_history)
    plt.xlabel('number of itterations')
    plt.ylabel('Cost function J')
    plt.show()
    
![alt text](data:image/png;base64,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%0A)

# Plot of fitting plan and the data


    from mpl_toolkits.mplot3d import Axes3D 
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np



    fig = plt.figure()
    ax=plt.axes(projection='3d')

    ax.scatter(x1, x2, y, c='b')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2 ')
    ax.set_zlabel('x3')
    n = 200

    Xs, Ys = np.meshgrid(x1, x2)
    Zs = np.array([np.matrix( (1,t0, t1) )*thetaSolution for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
    Zs = np.reshape(Zs, Xs.shape)

    fig = plt.figure(figsize=(7,7))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(' y ')
    ax.view_init(10, 10)
    ax.plot_surface(Xs, Ys, Zs, cmap=cm.jet,alpha=0.4)
    
![alt text](https://github.com/mohammedAljadd/2D-linear-regression/blob/main/plots/plan.PNG)

As you can see the plan is fitting to our data.
It's better to change the view angle to see the plan :

![alt text](https://github.com/mohammedAljadd/2D-linear-regression/blob/main/plots/plan_other.PNG)

 # Performance of regression 
 
 ![alt text](https://ashutoshtripathicom.files.wordpress.com/2019/01/rsquarecanva2.png)

 
 This factor should be close to 1.
 
    y_variance = len(y)*np.var(y)
    sum_squared_errors = (2*m)*cost(optimal_beta)
    Performance = 1 - ( sum_squared_errors )/(y_variance)
    print('The performance R is ',Performance) 
    
 The performance R is  0.993259187678495
