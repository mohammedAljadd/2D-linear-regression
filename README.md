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
    
![alt text](https://github.com/mohammedAljadd/2D-linear-regression/blob/main/plots/jitt.PNG)

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
