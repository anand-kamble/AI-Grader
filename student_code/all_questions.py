# Add import files
import pickle


# -----------------------------------------------------------
# https://www.studocu.com/sg/document/james-cook-university-singapore/information-security/exercise-3-soln-clustering/68793700
def question1():
    answers = {}

    # type: bool (True/False)
    answers["(a)"] = True

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "K means also takes into account the values of outliers which can produce some randomness."

    # type: bool (True/False)
    answers["(b)"] = True

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "Since we are eliminating the outliers, there will be no noise. And AHC will produce same results."

    # type: bool (True/False)
    answers["(c)"] = False

    # type: explanatory string (at least four words)
    # https://medium.com/@waziriphareeyda/difference-between-k-means-and-hierarchical-clustering-edfec55a34f8
    answers["(c) explain"] = "K-means is computationally intesive."

    # THIS IS NOT A TRUE FALSE QUESTION    
    # type: bool (True/False)
    answers["(d)"] = False

    # type: explanatory string (at least four words)
    answers["(d) explain"] = "Splitting decreases SSE because we have two centroids for the same set of points"

    # type: bool (True/False)
    answers["(e)"] = True

    # type: explanatory string (at least four words)
    answers["(e) explain"] = "For K-means, SSE is an inverse measure of the cohesion of clusters, and thus, as SSE decreases, cohesion increases and vice-versa"

    # type: bool (True/False)
    answers["(f)"] = True

    # type: explanatory string (at least four words)
    answers["(f) explain"] = "For K-means SSB is direct measure of the separation of clusters, and thus, as SSB increases, separation increases and vice-versa."

    # type: bool (True/False)
    answers["(g)"] = False

    # type: explanatory string (at least four words)
    answers["(g) explain"] = "Those are related since TSS = SSE + SSB."

    # type: bool (True/False)
    answers["(h)"] = True

    # type: explanatory string (at least four words)
    answers["(h) explain"] = "because TSS = SSE + SSB"

    # type: bool (True/False)
    answers["(i)"] = True

    # type: explanatory string (at least four words)
    answers["(i) explain"] = "cohesion and separation are inversely related. As cohesion increases, separation decreases and vice-versa."

    return answers


# -----------------------------------------------------------
def question2():
    answers = {}

    # type: bool (True/False)
    answers["(a)"] = True

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "Since d is greater than both r1 and r2, k-means will shift the centroid to the center os these two clusters. (Assuming the value of k >= 2 )"

    # type: bool (True/False)
    answers["(b)"] = False

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "Both centroids will have points from both clusters as they are very close to each other."

    # type: bool (True/False)
    answers["(c)"] = True

    # type: explanatory string (at least four words)
    answers["(c) explain"] = "For the 1st iteration, true. After some iterations, the centroid will move closer to the clusters."

    return answers


# -----------------------------------------------------------
def question3():
    answers = {}

    # ========================== (a) ==========================
    # type: a string that evaluates to a float
    # Assuming a value of R.
    # k = 4
    # R = 2 
    # SSE = k * R**2
    answers["(a) SSE"] = "4 * R**2"


    # ========================== (b) ==========================
    # To compute the total Sum of Squared Errors (SSE) of the data points to the origin (O), 
    # you can use the formula:

    # $$ SSE = \sum_{i=1}^{n} (x_i^2 + y_i^2) $$

    # where $(x_i, y_i)$ are the coordinates of each data point. Since the origin is (0, 0) 
    # and the distance from each data point to the origin is the Euclidean distance, 
    # which is the square root of the sum of squares of the coordinates, 
    # we can simplify the formula as:

    # $$ SSE = \sum_{i=1}^{n} (x_i^2 + y_i^2) = \sum_{i=1}^{n} R^2 = n \times R^2 $$


    # type: a string that evaluates to a float
    # a = 3 # Assuming a value of a 
    # b = 4 # Assuming a value of b
    # p1 = ( b , a + R)
    # p2 = ( b , a - R)
    # p3 = ( b + R , a)
    # p4 = ( b - R , a)

    # def SSE(points):
    #     ans = 0
    #     for point in points:
    #         ans += (point[0] - b)**2 + (point[1] - a)**2
    #     return ans
    
    # points = [p1, p2, p3, p4]
    answers["(b) SSE"] = " 4 * (a**2 + b**2 + c**2)"

    # ========================== (c) ==========================    
    
    # type: a string that evaluates to a float
    # Assuming that the point D is at (b, 0)
    # D = (b, 0)
    # u = ( b , R + (R/2))
    # v = ( b - (R/2) , R)
    # x = ( b, R/2)
    # w = ( b + (R/2) , R)

    # Since it is symmetric through the horizintal line running thourgh point D.

    # u1 = ( u[0], -u[1])
    # v1 = ( v[0], -v[1])
    # x1 = ( x[0], -x[1])
    # w1 = ( w[0], -w[1])

    # points = [u, v, x, w, u1, v1, x1, w1]
    answers["(c) SSE"] = "4*(R**2 + (R/2)**2)" #"2 * (4 * (R/2)^2 + 4 * R^2)"
    # print(answers["(c) SSE"])

    return answers


# -----------------------------------------------------------
def question4():
    answers = {}

    # type: int
    answers["(a) Circle (a)"] = 1

    # type: int
    answers["(a) Circle (b)"] = 1

    # type: int
    answers["(a) Circle (c)"] = 1

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "Distance between the clusters is same. Assuming that the distance is greater than the radius. and these cluster are not intersecting."

    # type: int
    answers["(b) Circle (a)"] = 1

    # type: int
    answers["(b) Circle (b)"] = 1

    # type: int
    answers["(b) Circle (c)"] = 1

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "One centroid which is in B, will move towards cluster C."

    # type: int
    answers["(c) Circle (a)"] = 0

    # type: int
    answers["(c) Circle (b)"] = 0

    # type: int
    answers["(c) Circle (c)"] = 2

    # type: explanatory string (at least four words)
    answers["(c) explain"] = "Point which is in A will move between A and B. The remaining two points will stay inside cluster C and divide it into two parts."

    return answers


# -----------------------------------------------------------
# https://www.chegg.com/homework-help/questions-and-answers/hierarchical-clustering-intermediate-stage-agglomerative-clustering-algorithm-given-three--q50235522
def question5():
    answers = {}

    # type: set
    answers["(a)"] = {'Group A', 'Group B'}

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "smallest single link distance between A (rightmost point) and B (Leftmost point) "

    # type: set
    answers["(b)"] = {'Group A', 'Group C'}

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "smallest complete link distance between A (rightmost point) and C (rightmost point) "

    return answers


# -----------------------------------------------------------
def question6():
    #https://becominghuman.ai/dbscan-clustering-algorithm-implementation-from-scratch-python-9950af5eed97
    def euclidean_distance(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    def range_query(data, point, eps):
        neighbors = []
        for i, neighbor in enumerate(data):
            if euclidean_distance(point, neighbor) <= eps:
                neighbors.append(i)
        return neighbors

    def dbscan(data, eps, min_samples):
        labels = [0] * len(data)
        core_points = []
        boundary_points = []
        noise_points = []

        for i, point in enumerate(data):
            if labels[i] != 0:
                continue
            neighbors = range_query(data, point, eps)
            if len(neighbors) < min_samples:
                noise_points.append(i)
                labels[i] = -1 
                continue
            labels[i] = i + 1
            core_points.append(i + 1)
            for neighbor_index in neighbors:
                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = i + 1
                    boundary_points.append(neighbor_index)
                elif labels[neighbor_index] == 0:
                    labels[neighbor_index] = i + 1
                    core_points.append(neighbor_index)
                    neighbor_neighbors = range_query(data, data[neighbor_index], eps)
                    if len(neighbor_neighbors) >= min_samples:
                        neighbors.extend(neighbor_neighbors)
                else:
                    boundary_points.append(neighbor_index)

        return labels, core_points, boundary_points, noise_points

    
    # Sample data
    data = [
        [0.0,0.0],
        [1.0,1.0],
        [1.0,2.0],
        [1.0,3.0],
        [2.0,2.0],
        [2.0,1.0],
        [3.0,1.0],
        [4.0,4.0],
        [5.0,5.0],
        [5.0,6.0],
        [6.0,6.0],
        [6.0,5.0]
    ]

    point_label = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M"]

    def get_label_by_point(point, data = data, labels = point_label):
        for i, data_point in enumerate(data):
            if data_point == point:
                return labels[i]
        return None
    
    def separate_by_cluster(cluster_labels, data):
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(data[i])
        return clusters
    
    # ========================== (a) ==========================

    # DBSCAN parameters
    eps = 1
    min_samples = 3

    # Run DBSCAN
    cluster_labels, core_points, boundary_points, noise_points = dbscan(data, eps, min_samples)

    answers = {}

    CORE = set([point_label[i] for i in core_points])
    BOUNDARY = set([point_label[i] for i in boundary_points])
    NOISE = set([point_label[i] for i in noise_points])


    # type: set
    answers["(a) core"] = {'J', 'B', 'C', 'L', 'E', 'I', 'F', 'M'}
    # type: set
    answers["(a) boundary"] = CORE.difference(BOUNDARY)

    # type: set
    answers["(a) noise"] = NOISE

    # ========================== (b) ==========================

    separated_clusters = separate_by_cluster(cluster_labels, data)

    separated_clusters.pop(-1,None)
    final_clusters = {}
    for cluster_id, points in separated_clusters.items():
        final_clusters[cluster_id] = []
        for p in separated_clusters[cluster_id]:
            final_clusters[cluster_id].append(get_label_by_point(p))

    # type: set
    answers["(b) cluster 1"] = set(list(final_clusters.items())[0][1])

    # type: set
    answers["(b) cluster 2"] = set(list(final_clusters.items())[1][1])

    # type: set
    answers["(b) cluster 3"] = set()

    # type: set
    answers["(b) cluster 4"] = set()

    # ========================== (c) ==========================    
    eps = 2**0.5
    min_samples = 3
    cluster_labels, core_points, boundary_points, noise_points = dbscan(data, eps, min_samples)

    CORE = set([point_label[i] for i in core_points])
    BOUNDARY = set([point_label[i] for i in boundary_points])
    NOISE = set([point_label[i] for i in noise_points])
    # type: set
    answers["(c)-a core"] = {'J', 'B', 'G', 'D', 'C', 'L', 'E', 'I', 'F', 'M'}

    # type: set
    answers["(c)-a boundary"] = {'A', 'H'}

    # type: set
    answers["(c)-a noise"] = set()

    separated_clusters = separate_by_cluster(cluster_labels, data)

    separated_clusters.pop(-1,None)
    final_clusters = {}
    for cluster_id, points in separated_clusters.items():
        final_clusters[cluster_id] = []
        for p in separated_clusters[cluster_id]:
            final_clusters[cluster_id].append(get_label_by_point(p))

    # type: set
    answers["(c)-b cluster 1"] = set(list(final_clusters.items())[0][1]).union(set(list(final_clusters.items())[1][1]))

    # type: set
    answers["(c)-b cluster 2"] = {'A'}

    # type: set
    answers["(c)-b cluster 3"] = set()

    # type: set
    answers["(c)-b cluster 4"] = set()

    print(" ===== answers =====")
    print(answers)
    return answers


# -----------------------------------------------------------
def question7():
    answers = {}
    """
    This wasn't necessary to solve the question.
    But I still implemented it to check my answers.

    def calculate_entropy(confusion_matrix):
        matrix = np.array(confusion_matrix)
        total_objects = np.sum(matrix)
        entropies = []
        
        for cluster in matrix:
            proportions = cluster / np.sum(cluster)
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))  
            entropies.append(entropy)
        return entropies

    confusion_matrix = [
        [10, 100, 20, 10, 30000],
        [3000, 10, 1000, 10, 0],
        [10, 3000, 500, 150, 200],
        [2000, 2500, 1500, 3000, 1400]
    ]

    entropies = calculate_entropy(confusion_matrix)
    print("Entropies for each cluster:", entropies)

    Output was:
    [0.04868479009392732, 0.8574416753624556, 1.0901763695276254, 2.261516521714887]
    """
    # type: string
    answers["(a)"] = "Cluster 4"

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "This cluster has similar number of objects in each class. "

    # type: string
    answers["(b)"] = "Cluster 1"

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "This cluster has the most of the objects in one class i.e. water."

    return answers


# -----------------------------------------------------------
def question8():
    answers = {}

    # type: string
    answers["(a) Matrix 1"] = "Dataset Z"

    # type: explanatory string (at least four words)
    answers["(a) explain diag entries, Matrix 1"] = "Color indicated distance is closer to zero, which means the points are dense and not spreaded."

    # type: explanatory string (at least four words)
    answers["(a) explain non-diag entries, Matrix 1"] = "If we look at (1,3) first column and third row, and (3,1) third column and first row, Colors are mixed, which can only happen in Dataset Z for clusters A and C."

    # type: string
    answers["(a) Matrix 2"] = "Dataset X"

    # type: explanatory string (at least four words)
    answers["(a) explain diag entries, Matrix 2"] = "2 Dark colored blocks indicate that 2 of the cluters are dense."

    # type: explanatory string (at least four words)
    answers["(a) explain non-diag entries, Matrix 2"] = "Color for A and D is red, which means A and D clusters are far from each other."

    # type: string
    answers["(a) Matrix 3"] = "Dataset Y"

    # type: explanatory string (at least four words)
    answers["(a) explain diag entries, Matrix 3"] = "2 Dark colored blocks indicate that 2 of the cluters are dense."

    # type: explanatory string (at least four words)
    answers["(a) explain non-diag entries, Matrix 3"] = "Red colored blocks indicate that B and D clusters are far from each other."

    # type: string
    answers["(b) Row 1"] = "Cluster A"

    # type: string
    answers["(b) Row 2"] = "Cluster B"

    # type: string
    answers["(b) Row 3"] = "Cluster C"

    # type: string
    answers["(b) Row 4"] = "Cluster D"

    # type: explanatory string (at least four words)
    answers["(b) Row 1 explain"] = "Color is not as dark as others meaning that it is not dense."

    # type: explanatory string (at least four words)
    answers["(b) Row 2 explain"] = "It is dark, meaning that is is dense and is almost equidistant to other 2 clusters."

    # type: explanatory string (at least four words)
    answers["(b) Row 3 explain"] = "It is dense since it has dark color, but it is not closer to 1st cluster."

    # type: explanatory string (at least four words)
    answers["(b) Row 4 explain"] = "Is is not dense and is far from the 1st clusters."

    return answers


# -----------------------------------------------------------
def question9():
    answers = {}

    # type: list
    answers["(a)"] = ["Hierarchical", "overlapping", "partial"]

    # type: list
    answers["(b)"] = ["Partitional", "exclusive", "complete"]

    # type: list
    answers["(c)"] = ["Partitional", "fuzzy", "complete"]

    # type: list
    answers["(d)"] = [ "Hierarchical", "overlapping", "partial"]

    # type: list
    answers["(e)"] = ["Partitional", "Exclusive", "partial"]

    # type: explanatory string (at least four words)
    answers["(e) explain"] = "Not every student in Computer science has taken Data mining."

    return answers


# -----------------------------------------------------------
def question10():
    answers = {}

    # type: string
    answers["(a) Figure (a)"] = "No"

    # type: string
    answers["(a) Figure (b)"] = "Yes"

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "DBSCAN prefers denser points and will not work well with the first figure."

    # type: string
    answers["(b) Figure (a)"] = "No"

    # type: string
    answers["(b) Figure (b)"] = "Yes"

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "K means will work but it will include all the points, which are the outliers."

    # type: string
    # answers["(c)"] = "Use DBSCAN with low radius (eps) , All the points that are outliers, mark them, then run DBSCAN again with those points. And then we will get the 4 required clusters."
    answers["(c)"] = "Take the reciprocal of the density as the new density and use DBSCAN."
    return answers


# --------------------------------------------------------
if __name__ == "__main__":
    answers_dict = {}
    answers_dict["question1"] = question1()
    answers_dict["question2"] = question2()
    answers_dict["question3"] = question3()
    answers_dict["question4"] = question4()
    answers_dict["question5"] = question5()
    answers_dict["question6"] = question6()
    answers_dict["question7"] = question7()
    answers_dict["question8"] = question8()
    answers_dict["question9"] = question9()
    answers_dict["question10"] = question10()
    print(" ==================== end code ====================")

    with open("answers.pkl", "wb") as f:
        pickle.dump(answers_dict, f)
