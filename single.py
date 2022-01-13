import math

import numpy as np
import matplotlib.pyplot as plt

def SRPT(actual, release):

    remainingActual = actual.copy()  # List with the remaining actual processing time of each job

    # Indices of active jobs (have arrived && haven't finished [actual] execution)
    indActive = [idx for idx, e in enumerate(release) if e == 0]

    # Future release times
    releaseTimes = set(release)

    if len(indActive) > 0:
        releaseTimes.remove(0)

    cost = 0
    time = 0

    while sum(remainingActual) > 0:
        # Execute the job with the shortest remaining actual processing time (between active jobs)
        if len(indActive) > 0:
            # Find the shortest such job
            ind = indActive[0]
            mini = remainingActual[ind]
            for e in indActive:
                if remainingActual[e] < mini:
                    ind = e
                    mini = remainingActual[e]

            # Augment time until a new job arrives OR the job finishes its [actual] execution
            if len(releaseTimes) > 0 and min(releaseTimes) - time < remainingActual[ind]:
            # A new job arrives
                # # Update the remaining processing times [actual] for the executed job
                remainingActual[ind] -= (min(releaseTimes) - time)

                # Update time
                time = min(releaseTimes)

                # Update active jobs AND future release times
                indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                releaseTimes.remove(min(releaseTimes))

            else:
            # The job finishes its execution
                # Update time AND the cost of the objective function
                time += remainingActual[ind]
                cost += time

                # # Update the remaining processing times for the executed job
                remainingActual[ind] = 0
                # Remove it from active jobs
                indActive.remove(ind)

                # Update active jobs AND future release times
                if len(releaseTimes) > 0 and min(releaseTimes) - time == 0:
                    indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                    releaseTimes.remove(min(releaseTimes))

        else:
        # There is no active job, we augment the time to the next release time
            if len(releaseTimes) == 0:  # All jobs have arrived
                # This case shouldn't exist as while condition would also be false
                return cost

            # Augment time to next release time
            time = min(releaseTimes)
            # Update active jobs AND future release times
            print("There is idle time in the schedule")
            indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
            releaseTimes.remove(min(releaseTimes))

    return cost


def SRPPT(pred, actual, release):

    remainingActual = actual.copy()  # List with the remaining actual processing time of each job
    remainingPred = pred.copy()  # List with the remaining predicted processing time of each job

    # Indices of active jobs (have arrived && haven't finished [actual] execution)
    indActive = [idx for idx, e in enumerate(release) if e == 0]

    # Future release times
    releaseTimes = set(release)

    if len(indActive) > 0:
        releaseTimes.remove(0)

    cost = 0
    time = 0

    while sum(remainingActual) > 0:
        # Execute the job with the shortest remaining predicted processing time (between active jobs)
        if len(indActive) > 0:
            # Find the shortest such job
            ind = indActive[0]
            mini = remainingPred[ind]
            for e in indActive:
                if remainingPred[e] < mini:
                    ind = e
                    mini = remainingPred[e]

            # Augment time until a new job arrives OR the job finishes its [actual] execution
            if len(releaseTimes) > 0 and min(releaseTimes) - time < remainingActual[ind]:
            # A new job arrives
                # Update the remaining processing times [actual and predicted] for the executed job
                remainingActual[ind] -= (min(releaseTimes) - time)
                remainingPred[ind] -= (min(releaseTimes) - time)

                # Update time
                time = min(releaseTimes)

                # Update active jobs AND future release times
                indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                releaseTimes.remove(min(releaseTimes))

            else:
            # The job finishes its [actual] execution
                # Update time AND the cost of the objective function
                time += remainingActual[ind]
                cost += time

                # Update the remaining processing times [actual and predicted] for the executed job
                remainingActual[ind] = 0
                remainingPred[ind] = 0
                # Remove it from active jobs
                indActive.remove(ind)

                # Update active jobs AND future release times
                if len(releaseTimes) > 0 and min(releaseTimes) - time == 0:
                    indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                    releaseTimes.remove(min(releaseTimes))

        else:
        # There is no active job, we augment the time to the next release time
            if len(releaseTimes) == 0:  # All jobs have arrived
                # This case shouldn't exist as while condition would also be false
                return cost

            # Augment time to next release time
            time = min(releaseTimes)
            # Update active jobs AND future release times
            print("There is idle time in the schedule")
            indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
            releaseTimes.remove(min(releaseTimes))

    return cost


def RR(actual, release):

    remainingActual = actual.copy()  # List with the remaining actual processing time of each job

    # Indices of active jobs (have arrived && haven't finished [actual] execution)
    indActive = [idx for idx, e in enumerate(release) if e == 0]

    # Future release times
    releaseTimes = set(release)

    if len(indActive) > 0:
        releaseTimes.remove(0)

    cost = 0
    time = 0

    while sum(remainingActual) > 0:
    # Execute the active jobs in parallel until one finishes or a new one arrives
        if len(indActive) > 0:
            # Find the shortest such job
            ind = indActive[0]
            mini = remainingActual[ind]
            for e in indActive:
                if remainingActual[e] < mini:
                    ind = e
                    mini = remainingActual[e]

            # Augment time until a new job arrives OR the shortest job finishes its [actual] execution
            if len(releaseTimes) > 0 and min(releaseTimes) - time < remainingActual[ind] * len(indActive):
            # A new job arrives
                # Update the remaining processing times [actual] for all active job
                for i in indActive:
                    remainingActual[i] -= (min(releaseTimes) - time) / len(indActive)

                # Update time
                time = min(releaseTimes)

                # Update active jobs AND future release times
                indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                releaseTimes.remove(min(releaseTimes))

            else:
            # The shortest job finishes its [actual] execution
                # Update time AND the cost of the objective function
                time += remainingActual[ind] * len(indActive)
                cost += time

                # Update the remaining processing times [actual] for all active job
                elapsed = remainingActual[ind]
                for i in indActive:
                    remainingActual[i] -= elapsed

                # Remove the shortest job that finished execution from active jobs
                indActive.remove(ind)

                # Update active jobs AND future release times
                if len(releaseTimes) > 0 and min(releaseTimes) - time == 0:
                    indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                    releaseTimes.remove(min(releaseTimes))

        else:
        # There is no active job, we augment the time to the next release time
            if len(releaseTimes) == 0:  # All jobs have arrived
                # This case shouldn't exist as while condition would also be false
                return cost

            # Augment time to next release time
            time = min(releaseTimes)
            # Update active jobs AND future release times
            print("There is idle time in the schedule")
            indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
            releaseTimes.remove(min(releaseTimes))

    return cost


def preferential(pred, actual, release, lamb):

    remainingActual = actual.copy()  # List with the remaining actual processing time of each job
    remainingPred = pred.copy()  # List with the remaining predicted processing time of each job

    # Indices of active jobs (have arrived && haven't finished [actual] execution)
    indActive = [idx for idx, e in enumerate(release) if e == 0]

    # Future release times
    releaseTimes = set(release)

    if len(indActive) > 0:
        releaseTimes.remove(0)

    cost = 0
    time = 0

    while sum(remainingActual) > 0:
        # Execute the active jobs in parallel until one finishes or a new one arrives
        if len(indActive) > 0:
            # The index of the shortest job according to the actual processing times
            ind = indActive[0]
            mini = remainingActual[ind]
            # The index of the shortest job according to the predicted processing times
            indPred = indActive[0]
            miniPred = remainingPred[indPred]
            for e in indActive:
                if remainingPred[e] < miniPred:
                    indPred = e
                    miniPred = remainingPred[e]
                if remainingActual[e] < mini:
                    ind = e
                    mini = remainingActual[e]

            # Find which job will finish first
            # The job executed by SRPPT will finish at time remainingActual[indPred]/(lamb + (1-lamb)/len(indActive))
            # The shortest job executed by RR will finish at time remainingActual[ind]*len(indActive)/(1-lamb)
            timeRR = remainingActual[ind] * len(indActive) / (1 - lamb)
            timeSRPPT = remainingActual[indPred] / (lamb + (1 - lamb) / len(indActive))

            if ind == indPred:
                finishedJob = ind
                finishingTime = timeSRPPT
            else:
                if timeRR < timeSRPPT:
                    finishedJob = ind
                    finishingTime = timeRR
                else:
                    finishedJob = indPred
                    finishingTime = timeSRPPT

            # Augment time until a new job arrives OR the shortest job finishes its [actual] execution
            if len(releaseTimes) > 0 and min(releaseTimes) - time < finishingTime:
            # A new job arrives
                # Update the remaining processing times [actual and predicted] for all active job
                for i in indActive:
                    remainingActual[i] -= (1 - lamb) * (min(releaseTimes) - time) / len(indActive)
                    remainingPred[i] -= (1 - lamb) * (min(releaseTimes) - time) / len(indActive)

                remainingActual[indPred] -= (lamb) * (min(releaseTimes) - time)
                remainingPred[indPred] -= (lamb) * (min(releaseTimes) - time)

                # Update time
                time = min(releaseTimes)

                # Update active jobs AND future release times
                indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                releaseTimes.remove(min(releaseTimes))

            else:
            # The shortest job finishes its [actual] execution
                # Update time AND the cost of the objective function
                time += finishingTime
                cost += time

                # Update the remaining processing times [actual and predicted] for all active job
                for i in indActive:
                    remainingActual[i] -= (1 - lamb) * finishingTime / len(indActive)
                    remainingPred[i] -= (1 - lamb) * finishingTime / len(indActive)

                remainingActual[indPred] -= (lamb) * finishingTime
                remainingPred[indPred] -= (lamb) * finishingTime

                remainingActual[finishedJob] = 0  # To avoid rounding problems

                # Remove the shortest job that finished execution from active jobs
                indActive.remove(finishedJob)

                # Update active jobs AND future release times
                if len(releaseTimes) > 0 and min(releaseTimes) - time == 0:
                    indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
                    releaseTimes.remove(min(releaseTimes))

        else:
        # There is no active job, we augment the time to the next release time
            if len(releaseTimes) == 0:  # All jobs have arrived
                # This case shouldn't exist as while condition would also be false
                return cost

            # Augment time to next release time
            time = min(releaseTimes)
            # Update active jobs AND future release times
            print("There is idle time in the schedule")
            indActive += [idx for idx, e in enumerate(release) if e == min(releaseTimes)]
            releaseTimes.remove(min(releaseTimes))

    return cost


def simulations(numOfJobs=50, numOfPoints=2000, lamb=1/2, sigma=1, tau=1):
    alpha = 1.1
    actual_pred_release = []
    for i in range(numOfPoints):
        # actual processing time
        actual_pareto = [round(np.random.pareto(alpha), 2) * 100 + 1 for i in range(numOfJobs)]
        avg = sum(actual_pareto)/numOfJobs

        # error
        noise_normal = []
        for i in range(numOfJobs):
            noise_normal.append(round(np.random.randn() * sigma, 2))

        # predicted processing time
        pred_normal_pareto = [
            round(actual_pareto[i] + noise_normal[i]) if round(actual_pareto[i] + noise_normal[i], 2) > 0 else 0 for i
            in range(numOfJobs)]

        # release
        size = sum(actual_pareto)

        # poisson = np.random.poisson(2, numOfJobs)
        # release_dates = []
        # for i in range(numOfJobs):
        #     if i == 0:
        #         release_dates.append(poisson[i])
        #     else:
        #         release_dates.append(release_dates[i-1] + poisson[i] * avg)

        release_dates = [np.random.uniform(0, tau * size) for i in range(numOfJobs)]

        data = (actual_pareto, pred_normal_pareto, release_dates)
        actual_pred_release.append(data)

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

    for a_p_r in actual_pred_release:
        actual, pred, release = a_p_r

        # Run the algorithms
        algSRPPT = SRPPT(pred, actual, release)
        algRR = RR(actual, release)
        algPRR = preferential(pred, actual, release, lamb)
        opt = SRPT(actual, release)

        # Calculate the ratio
        ratioSRPPT.append(algSRPPT / opt)
        ratioRR.append(algRR / opt)
        ratioPRR.append(algPRR / opt)

    # Keep the average of all 'numOfPoints' executions
    Y.append(sum(ratioSRPPT)/len(ratioSRPPT))
    Y.append(sum(ratioRR)/len(ratioRR))
    Y.append(sum(ratioPRR)/len(ratioPRR))

    return Y

def competitive2error():
    X = []
    Y = []
    for s in range(0, 5000, 50):
        X.append(s)
        Y.append(simulations(sigma=s, tau=0.1))

    plt.plot(X, Y, label=["SRPPT", "RR", "PRR"])
    plt.xlabel('Prediction Error')
    plt.ylabel('Competitive Ratio')
    plt.legend(loc='upper left')
    plt.show()

def competitive2release():
    X = []
    Y = []

    for t in range(0, 18, 1):
        X.append(t/10)
        Y.append(simulations(sigma=1000, tau=t/10)[2])

    plt.plot(X, Y, color='C2', label="PRR")
    plt.xlabel(r'$\tau$')
    plt.ylabel('Competitive Ratio')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':

    # competitive2error()

    competitive2release()

    # predicted = [5, 3, 2, 1, 1]
    # actual = [2, 3, 2, 5, 1]
    # release = [0, 0, 5, 5, 9]
    # completionTimes = [0, 0, 0, 0, 0]
    #
    # print("\nSRPT (optimal) : ")
    # print(SRPT(actual, release))
    #
    # print("\nSRPPT : ")
    # print(SRPPT(predicted, actual, release))
    #
    # print("\nRound-Robin :")
    # print(RR(actual, release))
    #
    # print("\nPreferential :")
    # print(preferential(predicted, actual, release, 1 / 2))