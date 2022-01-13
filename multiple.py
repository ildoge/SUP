import numpy as np
import matplotlib.pyplot as plt


def sumOfCompletions(list):
    time = 0
    objective = 0
    for i in list:
        time += i
        objective += time

    return objective


def orderInd(list):
    ordered = [i for i in range(len(list))]
    ordered.sort(key=lambda x: list[x])

    return ordered


def SPT_m(actual, machines):

    order = sorted(actual)
    schedule = {}

    for i in range(0, len(order)):
        if i < machines:
            schedule[i % machines] = [order[i]]
        else:
            schedule[i % machines].append(order[i])

    cost = []
    for i in range(0, machines):
        cost.append(sumOfCompletions(schedule[i]))

    return sum(cost)


def SPPT_m(pred, actual, machines):

    order = [actual[i] for i in orderInd(pred)]
    schedule = {}

    for i in range(0, len(order)):
        if i < machines:
            schedule[i % machines] = [order[i]]
        else:
            schedule[i % machines].append(order[i])

    cost = []
    for i in range(0, machines):
        cost.append(sumOfCompletions(schedule[i]))

    return sum(cost)


def RR_m(actual, machines, completions=[]):

    remainingActual = actual.copy()

    indActive = [i for i in range(len(actual))]

    cost = 0
    time = 0

    while sum(remainingActual) > 0:
        #
        if len(indActive) >= machines:  #
            #
            ind = indActive[0]
            mini = remainingActual[ind]
            for e in indActive:
                if remainingActual[e] < mini:
                    ind = e
                    mini = remainingActual[e]

            #
            time += remainingActual[ind] * len(indActive) / machines
            cost += time
            #
            if completions != []:
                completions[ind] = time

            #
            elapsed = remainingActual[ind]
            for i in indActive:
                remainingActual[i] -= elapsed

            #
            indActive.remove(ind)

        else:
            for i in indActive:
                t = time + remainingActual[i]
                cost += t
                if completions != []:
                    completions[i] = t
                remainingActual[i] = 0
                indActive.remove(i)

    return cost


def preferential_m(pred, actual, machines, lamb):

    remainingActual = actual.copy()

    # order = [actual[i] for i in orderInd(pred)]
    orderIndex = orderInd(pred)

    activeIndex = []

    for i in range(0, len(orderIndex)):
        if i < machines:
            activeIndex.append([orderIndex[i]])
        else:
            activeIndex[i%machines].append(orderIndex[i])

    cost = 0
    time = 0

    while sum(remainingActual) > 0:
        check = True

        activeJobs = 0
        for x in activeIndex:
            activeJobs += len(x)

        if activeJobs > machines:
            # Find the minimum job executed by SPPT (this is the smallest among the first jobs of each machine)
            # Initialize values for the search
            for x in range(0, len(activeIndex)):
                if activeIndex[x]:
                    indPred = activeIndex[x][0]
                    machinePred = x
            # Search for the shortest element that runs first on a machine
            miniPred = remainingActual[indPred]
            for i in range(0, len(activeIndex)):
                if activeIndex[i]:
                    if remainingActual[activeIndex[i][0]] < miniPred:
                        indPred = activeIndex[i][0]
                        miniPred = remainingActual[activeIndex[i][0]]
                        machinePred = i

            # Find the shortest job executed by RR (this is the smallest among the remainingActual)
            # Initialize values for the search
            for i in range(0, len(activeIndex)):
                if activeIndex[i]:
                    ind = activeIndex[i][0]
                    machine = i
                    job = 0
            mini = remainingActual[ind]
            # Search for the shortest element
            for i in range(0, len(activeIndex)):
                for j in range(0, len(activeIndex[i])):
                    if remainingActual[activeIndex[i][j]] < mini:
                        ind = activeIndex[i][j]
                        mini = remainingActual[activeIndex[i][j]]
                        machine = i
                        job = j

            # Find which job will finish first
            timeRR = mini * activeJobs / (1 - lamb) * machines
            timeSPPT = miniPred / (lamb + ((1 - lamb) * machines) / activeJobs)

            if machinePred == machine and job == 0:
                finishedJob = ind
                finishingTime = timeSPPT
            else:
                if timeRR < timeSPPT:
                    finishedJob = ind
                    finishingTime = timeRR
                    check = False
                else:
                    finishedJob = indPred
                    finishingTime = timeSPPT

            # Update time AND the cost of the objective function
            time += finishingTime
            cost += time

            # Update the remaining time of the jobs.
            # For all jobs, reduce the remaining time by the elapsed RR
            for e in activeIndex:
                for j in range(0, len(e)):
                    # remainingActual[e[j]] -= timeRR
                    reduce = (1 - lamb) * finishingTime * machines / activeJobs
                    if remainingActual[e[j]] - reduce > 0:
                        remainingActual[e[j]] -= reduce
                    else:
                        remainingActual[e[j]] = 0


            # How much they are executed by SPPT
            for i in activeIndex:
                if i:
                    remainingActual[i[0]] -= (lamb) * finishingTime
            remainingActual[finishedJob] = 0  # To avoid rounding errors

            if check:
                # If job finishes due to SPPT
                activeIndex[machinePred].pop(0)
            else:
                # If job finishes due to RR
                activeIndex[machine].pop(job)
        else:
            # We have as many jobs as machines (or less). We cannot run RR in parallel
            unfinished = []
            for x in activeIndex:
                for job in x:
                    unfinished.append(job)

            for i in range(0, len(unfinished)):
                t = time + remainingActual[unfinished[i]]
                cost += t
                remainingActual[unfinished[i]] = 0

    return cost


def simulations(machines=2, numOfJobs=250, numOfPoints=2000, lamb=1/2, sigma=1):
    alpha = 1.1
    actual_pred = []
    for i in range(numOfPoints):
        # actual processing time
        actual_pareto = [round(np.random.pareto(alpha), 2) * 100 + 1 for i in range(numOfJobs)]

        # error
        noise_normal = []
        for i in range(numOfJobs):
            noise_normal.append(round(np.random.randn() * sigma, 2))

        # predicted processing time
        pred_normal_pareto = [
            round(actual_pareto[i] + noise_normal[i]) if round(actual_pareto[i] + noise_normal[i], 2) > 0 else 0 for i
            in range(numOfJobs)]

        data = (actual_pareto, pred_normal_pareto)
        actual_pred.append(data)

        # print("----------- Input Data --------------")
        # print("Max size: ", max(actual_pareto))
        # print("Min size: ", min(actual_pareto))
        # print("Avg size: ", sum(actual_pareto)/len(actual_pareto))
        # print("Max noise: ", max(noise_normal))
        # print("Min noise: ", min(noise_normal))
        # print("Avg noise: ", sum(noise_normal)/len(noise_normal))
        # print("Max release: ", max(release_dates))
        # print("Min release: ", min(release_dates))
        # print("-------------------------------------")

    # actual_pred_release.sort(key=lambda apr: calcError(apr[1], apr[0]))
    Y = []
    ratioSRPPT = []
    ratioRR = []
    ratioPRR = []

    for a_p in actual_pred:
        actual, pred = a_p

        # Run the algorithms
        algSPPT_m = SPPT_m(pred, actual, machines)
        algRR_m = RR_m(actual, machines)
        algPRR_m = preferential_m(pred, actual, machines, lamb)
        opt = SPT_m(actual, machines)

        # Calculate the ratio
        ratioSRPPT.append(algSPPT_m / opt)
        ratioRR.append(algRR_m / opt)
        ratioPRR.append(algPRR_m / opt)

    # Keep the average of all 'numOfPoints' executions
    Y.append(sum(ratioSRPPT)/len(ratioSRPPT))
    Y.append(sum(ratioRR)/len(ratioRR))
    Y.append(sum(ratioPRR)/len(ratioPRR))

    return Y


if __name__ == '__main__':

    X = []
    Y = []
    for s in range(0, 5000, 50):
        X.append(s)
        Y.append(simulations(machines=5, sigma=s))

    plt.plot(X, Y, label=["SPPT(m)", "RR(m)", "PRR(m)"])
    plt.xlabel('Prediction Error')
    plt.ylabel('Competitive Ratio')
    plt.legend(loc='upper left')
    plt.show()


    # machines = 1
    # lamb = 1/2
    # predicted = [5, 3, 2, 1, 1]
    # actual = [2, 3, 2, 5, 1]
    # completionTimes = [0, 0, 0, 0, 0]
    #
    # print("\nSPT_m (optimal) : ")
    # print(SPT_m(actual, machines))
    #
    # print("\nSPPT_m : ")
    # print(SPPT_m(predicted, actual, machines))
    #
    # print("\nRound-Robin_m :")
    # print(RR_m(actual, machines, completions=completionTimes))
    # # print("The completion times of the jobs are : ", completionTimes)
    #
    # print("\nPreferential :")
    # print(preferential_m(predicted, actual, machines, lamb))