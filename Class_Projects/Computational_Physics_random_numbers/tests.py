#This script defines the random number generators I tested as well as the tests
#used to analyze these random number generators

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import time

#defines our logistic map, returns uniform, normalized numbers
#xi+1 = 4*xi(1-xi)
def logistic(u,n,x0):
    #our arrays to fill, we want n uniform values and we take every
    #6th value from the "bad randoms", so bad randoms is 6x the length
    bad_randoms = np.zeros(int(n)*6)
    bad_randoms[0] = x0
    good_randoms = np.zeros(int(n))
    uniform_randoms = np.zeros(int(n))

    #loop through n*6
    for i in range(1,int(n)*6):
        #gets the previous number, applies the iterative map
        x = bad_randoms[i-1]
        num = u*x*(1-x)
        bad_randoms[i] = num

        #keep every 6th as indicated in the book
        if (i%6 == 0):
            good_randoms[int(i/6)] = num
    #normalize it
    good_randoms = good_randoms/np.max(good_randoms)

    #make them uniform
    for i in range(int(n)):

        num = good_randoms[i]
        uniform_randoms[i] = np.arccos(1-2*num)/np.pi

    #get rid of the zeros we initialized with
    good_randoms = np.delete(good_randoms,0)
    uniform_randoms = np.delete(uniform_randoms,0)
    return bad_randoms, good_randoms, uniform_randoms

#defines our logistic map, returns uniform, normalized numbers
#xi+1 = xi*exp(u(1-xi))
def ecology(u,n,x0):
    #our arrays to fill, we want n uniform values and we take every
    #6th value from the "bad randoms", so bad randoms is 6x the length
    bad_randoms = np.zeros(int(n*6))
    bad_randoms[0] = x0
    good_randoms = np.zeros(int(n))
    uniform_randoms = np.zeros(int(n))

    #loop through n*6
    for i in range(1,int(n*6)):
        #gets the previous number, applies the iterative map
        x = bad_randoms[i-1]
        num = x*np.exp(u*(1-x))
        bad_randoms[i] = num

        #keep every 6th as indicated in the book
        if (i%6 == 0):
            good_randoms[int(i/6)] = num

    #normalize it
    good_randoms = good_randoms/np.max(good_randoms)

    #make them uniform
    for i in range(int(n)):

        num = good_randoms[i]
        uniform_randoms[i] = np.arccos(1-2*num)/np.pi

    #get rid of the zeros we initialized with
    good_randoms = np.delete(good_randoms,0)
    uniform_randoms = np.delete(uniform_randoms,0)
    return bad_randoms, good_randoms, uniform_randoms

#our created random number generator which gets random numbers
#from timing addition and multiplication, extracting a random number
def rand_time(n):
    #our arrays to fill
    randomnumbers = np.array(())
    uniform_randoms = np.zeros(int(n))

    #loops through the array, gets random number and appends it
    for i in range(int(n)):
        #gets start time
        begin = time.time()
        x = 0
        y=2
        #does an operation
        for i in range(100):
            x = x + 1
            y**4
        #gets the end time
        end = time.time()

        #grabs a few digits from the first few decimals and a few digits from the last few
        #decimals and multiplies them to get a random number
        taken = str(float(end-begin))
        randn = float(taken[3:7])*(float(taken[7:11]))

        randomnumbers = np.append(randomnumbers,randn)
    #normlize
    randomnumbers = randomnumbers/np.max(randomnumbers)

    #make them uniform
    for i in range(int(n)):
        num = randomnumbers[i]
        uniform_randoms[i] = np.arccos(1-2*num)/np.pi

    return uniform_randoms

#generates an array of N's used in our tests to examine how the algorithm
#depends on the number of random values used
def makeN(maxn):
    i = 1
    maxn = maxn
    Ns = np.array(())
    #gets N values of 1,2,3...10,20,30...100,200,300...1000,2000,3000...10000...
    while i < maxn:
        if i < 10:
            i = i + 1
        elif i < 100 and i >= 10:
            i = i + 10
        elif i < 1000 and i >= 100:
            i = i + 100
        elif i < 10000 and i >= 1000:
            i = i + 1000
        elif i < 100000 and i >= 10000:
            i = i + 10000
        elif i < 1000000 and i >= 100000:
            i = i + 100000
        Ns = np.append(Ns,int(i))
    return Ns

#defines an instance of plot test 2, just plots the random values
def plot_test2(numbers):
    plt.figure(figsize = (12,8))

    plt.gca().spines['bottom'].set_linewidth(4)
    plt.gca().spines['left'].set_linewidth(4)
    plt.gca().spines['top'].set_linewidth(4)
    plt.gca().spines['right'].set_linewidth(4)
    csfont = {'fontname':'Verdana'}

    plt.tick_params(labelsize=28,pad = 10,length=15,width=4,which = "major")
    plt.tick_params(labelsize=28,pad = 10,length=12,width=2,which = "minor")
    plt.xlabel("i",**csfont,fontsize = 31,labelpad = 8)
    plt.ylabel("xi",**csfont,fontsize = 31,labelpad = 8)

    xs = np.arange(1,len(numbers)+1)
    plt.scatter(xs,numbers,color = "#d95f02",s = 50)

#runs multiple plot test 2's with different Ns
def run_plot_test2(u,x0,maptype):

    Ns = np.array((10,100,500,1000,5000,10000,50000,100000))

    for n in Ns:

        if maptype == "logistic":
            bad,good,uniform = logistic(u,n,x0)

        if maptype == "numpy":
            uniform = np.random.uniform(0,1,int(n))

        if maptype == "ecology":
            bad,good,uniform = ecology(u,n,x0)

        if maptype == "time":
            uniform = rand_time(n)

        plot_test2(uniform)

#defines one instance of plot test 5,plots the current random number and the next
#random number
def plot_test5(numbers,u,x0,savename,save=False):

    x = np.array(())
    y = np.array(())

    for i in range(int(len(numbers)/2)):
        x = np.append(x,numbers[2*i])
        y = np.append(y,numbers[2*i+1])

    plt.figure(figsize = (12,8))
    plt.gca().spines['bottom'].set_linewidth(4)
    plt.gca().spines['left'].set_linewidth(4)
    plt.gca().spines['top'].set_linewidth(4)
    plt.gca().spines['right'].set_linewidth(4)
    csfont = {'fontname':'Verdana'}

    plt.tick_params(labelsize=28,pad = 10,length=15,width=4,which = "major")
    plt.tick_params(labelsize=28,pad = 10,length=12,width=2,which = "minor")
    plt.xlabel("r2i",**csfont,fontsize = 31,labelpad = 8)
    title = "u = " + str(np.round(u,2)) + ", x0 = " + str(np.round(x0,2))
    plt.title(title,**csfont, fontsize = 30,y=1.02)
    plt.tight_layout()
    plt.ylabel("r2i+1",**csfont,fontsize = 31,labelpad = 8)
    plt.scatter(x,y,color = "#d95f02",s = 5)
    plt.tight_layout()

    if save == True:
        plt.savefig(savename,dpi=400)

    plt.show()

#runs many instances of plot test 5 with different Ns
def run_plot_test5(u,x0,maptype):

    Ns = np.array((10,100,500,1000,5000,10000,50000,100000))

    for n in Ns:

        if maptype == "logistic":
            bad,good,uniform = logistic(u,n,x0)

        if maptype == "numpy":
            uniform = np.random.uniform(0,1,int(n))

        if maptype == "ecology":
            bad,good,uniform = ecology(u,n,x0)

        if maptype == "time":
            uniform = rand_time(n)

        plot_test5(uniform,u,x0,0)

#defines an instance of test3
def test3(numbers, k):
    N = len(numbers)
    summ = np.sum(numbers**k)
    return summ/N, 1/(k+1), np.abs(summ/N - (1/(k+1))), 1/(np.sqrt(N)), N

#runs test 3 for an arranged differing Ns
def do_test3(u,x0,maptype,nmax=100000):
    deviations = np.array(())
    summs = np.array(())

    Ns = makeN(nmax)
    #generates random numbers for different Ns
    for n in Ns:
        if maptype == "logistic":
            bad,good,uniform = logistic(u,n,x0)

        if maptype == "numpy":
            uniform = np.random.uniform(0,1,int(n))

        if maptype == "ecology":
            bad,good,uniform = ecology(u,n,x0)

        if maptype == "time":
            uniform = rand_time(n)

        #for each N stores the summ, the deviation of the sum from the expected
        summ, one_o_k, difference, sqrtN, N = test3(uniform,4)

        deviations = np.append(deviations,difference)
        summs = np.append(summs,summ)

    #puts our values in log form for easier slope calculating
    logdev = np.log10(deviations)
    logn = np.log10(Ns)

    slope_plot(logn,logdev,3)

    print("These should be close to ",one_o_k)
    print(summs)

    #gets our slope
    slope,intercept, r, p, std_err = stats.linregress(logn,logdev)
    print("this slope should be close to -.5 it is equal to ",slope)

#defines an instance of test 4
def test4(numbers,k):
    N = len(numbers)
    summ = 0
    for i in range(N-k):
        prod = numbers[i]*numbers[i+k]
        summ += prod
    return summ/N, 1/4, np.abs(summ/N - (1/4)), 1/(np.sqrt(N)), N

#runs test 4 for an arranged differing Ns
def do_test4(u,x0,maptype,nmax=100000):
    deviations = np.array(())
    summs = np.array(())

    Ns = makeN(nmax)
    #generates random numbers for different Ns
    for n in Ns:
        if maptype == "logistic":
            bad,good,uniform = logistic(u,n,x0)

        if maptype == "numpy":
            uniform = np.random.uniform(0,1,int(n))

        if maptype == "ecology":
            bad,good,uniform = ecology(u,n,x0)

        if maptype == "time":
            uniform = rand_time(n)

        #for each N stores the summ, the deviation of the sum from the expected
        summ, one_fourth, difference, sqrtN, N = test4(uniform,4)

        deviations = np.append(deviations,difference)
        summs = np.append(summs,summ)

    #puts our values in log form for easier slope calculating
    logdev = np.log10(deviations)
    logn = np.log10(Ns)

    slope_plot(logn,logdev,4)

    print("These should be close to ",one_fourth)
    print(summs)

    #gets our slope
    slope,intercept, r, p, std_err = stats.linregress(logn,logdev)
    print("this slope should be close to -.5 it is equal to ",slope)

#performs test 6
def do_test6(u,x0,maptype):
    ks = np.array((1,3,7))
    Ns = np.array((100,1000,10000,100000))
    results = np.array([])

    passed = True

    #looping through different Ns, ks
    for n in Ns:
        k_batch = np.array(())
        for k in ks:

            if maptype == "logistic":
                bad,good,nums = logistic(u,n,x0)

            if maptype == "numpy":
                nums = np.random.uniform(0,1,n)

            if maptype == "ecology":
                bad,good,nums = ecology(u,n,x0)

            if maptype == "time":
                nums = rand_time(n)

            #calculates our summation
            summ = (np.sum(nums**k))/n
            sub = (1/(k+1))
            sqrtN = np.sqrt(n)
            result_k = sqrtN*np.abs(summ-sub)
            print(result_k, " k = ",k, "N = ", n)

            #determines whether our summation is accurate or not, is it of order 1?
            if np.abs(result_k) > 1:
                passed = False
                print("test failed with k = ", k,"N = ",n, "with a value of ",result_k)

            k_batch = np.append(k_batch,result_k)
        results = np.append(results,k_batch)
    if passed == False:
        print("")
        print("The test failed")

    if passed == True:
        print("")
        print("The test passed")
    return results

#our plots for tests 3 and 4
def slope_plot(numbers,Ns,testnum):
    plt.figure(figsize = (12,8))

    plt.gca().spines['bottom'].set_linewidth(4)
    plt.gca().spines['left'].set_linewidth(4)
    plt.gca().spines['top'].set_linewidth(4)
    plt.gca().spines['right'].set_linewidth(4)
    csfont = {'fontname':'Verdana'}

    plt.tick_params(labelsize=28,pad = 10,length=15,width=4,which = "major")
    plt.tick_params(labelsize=28,pad = 10,length=12,width=2,which = "minor")
    plt.xlabel("log(N)",**csfont,fontsize = 31,labelpad = 8)
    plt.ylabel("log(deviation)",**csfont,fontsize = 31,labelpad = 8)


    plt.plot(numbers,Ns,color = "#d95f02")

    title = "test " + str(testnum) + " error as function of N"
    plt.title(title,**csfont, fontsize = 30,y=1.02)


    plt.show()
