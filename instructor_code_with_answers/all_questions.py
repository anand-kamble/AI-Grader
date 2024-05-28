# Add import files
import pickle



# -----------------------------------------------------------
def question1():
    answers = {}


    # type: bool (True/False)
    answers["(a)"] = True

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "To be determined."

    # type: bool (True/False)
    answers["(b)"] = True

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "There is no random element in the algorithms for agglomerative  hierarchical techniques unless there are ties in the proximity values"

    # type: bool (True/False)
    answers["(c)"] = False

    # type: explanatory string (at least four words)
    answers["(c) explain"] = "Although k-means is more computationally efficient than agglomerative  hierarchical clustering, there are more efficient algorithms possible,  e.g., the leader algorithm. (See Exercise 12, Chapter 7.)"

    # type: bool (True/False)
    answers["(d)"] = False

    # type: explanatory string (at least four words)
    answers["(d) explain"] = "Splitting decreases SSE because we have two centroids  for the same set of points, which will reduce the distance  of points to the nearest centroid."

    # type: bool (True/False)
    answers["(e)"] = True

    # type: explanatory string (at least four words)
    answers["(e) explain"] = "For K-means, SSE is an inverse measure of the cohesion of clusters,  and thus, as SSE decreases, cohesion increases and vice-versa."

    # type: bool (True/False)
    answers["(f)"] = True

    # type: explanatory string (at least four words)
    answers["(f) explain"] = "For K-means SSB (the between sum of squares) is a direct  measure of the separation of clusters, and thus, as SSB  increases, separation increases and vice-versa."

    # type: bool (True/False)
    answers["(g)"] = False

    # type: explanatory string (at least four words)
    answers["(g) explain"] = "Cohesion and separation are independent for K-Means, i.e.,  improving cohesion (smaller SSE) doesn’t necessarily  improve separation (larger between sum of squares (SSB))"

    # type: bool (True/False)
    answers["(h)"] = True

    # type: explanatory string (at least four words)
    answers["(h) explain"] = "For K-means, the total sum of squares (TSS) is the sum of SSE  (or within cluster sum of squares) and the between sum of  squares (SSB), TSS is constant during the K-means clustering process.  See the book, page 577"

    # type: bool (True/False)
    answers["(i)"] = True

    # type: explanatory string (at least four words)
    answers["(i) explain"] = "For K-means, the total sum of squares (TSS) is the sum of  SSE (the within cluster sum of squares) and the between sum  of squares (SSB). Note TSS is constant at every step of the  K-means clustering process. See the book, page 577.   SSE is an inverse measure of cluster cohesion, while SSB is  a direct measure of cluster separation. Thus, as cohesion  increases, SSE decreases, and SSB (separation) increases since  TSS = SSE + SSB is a constant. When SSE is at a local minima,  BSS is at a local maxima."
    return answers




# -----------------------------------------------------------
def question2():
    answers = {}


    # type: bool (True/False)
    answers["(a)"] = True

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "The clusters are too far away for one centroid to  attract points from another."

    # type: bool (True/False)
    answers["(b)"] = False

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "The final clusters will have points from both of the two  shaded regions since they are close to each other and not  of circular shape."

    # type: bool (True/False)
    answers["(c)"] = True

    # type: explanatory string (at least four words)
    answers["(c) explain"] = "The centroid at 12.5 is farther away from all points than  any other clusters and will become empty."
    return answers




# -----------------------------------------------------------
def question3():
    answers = {}


    # type: a string that evaluates to a float
    answers["(a) SSE"] = "4 * R**2"

    # type: a string that evaluates to a float
    answers["(b) SSE"] = "4 * (a**2 + b**2 + c**2)"

    # type: a string that evaluates to a float
    answers["(c) SSE"] = "4*(R**2 + (R/2)**2)"
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
    answers["(a) explain"] = "All of circle A’s points will be assigned to the centroid in A.  About 1/3 of circle B’s points (the ones in the left third of circle  B) will be assigned to the centroid on the left in circle B.  The remaining 2/3 of the points in B and all the points in C will  be assigned to the centroid in the center of B. This will cause the  right centroid in B to move to circle C since C has many more points  than B.  In the next iteration, all points in A,B, and C will be  assigned to the centroids located in their own circles and K-means  will converge."

    # type: int
    answers["(b) Circle (a)"] = 1

    # type: int
    answers["(b) Circle (b)"] = 1

    # type: int
    answers["(b) Circle (c)"] = 1

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "Since circles A and B are close together and quite far away from  circle C, the points from both A and B will be assigned to the  centroid that is in A. The points in C will be split between the  two centroids in C, with each centroid having 50,000 points.  Since  A and B have the same number of points each, the centroid in A will  move between A and B. The centroids in C will move apart slightly  but both will remain in C, each having half of C’s points."

    # type: int
    answers["(c) Circle (a)"] = 0

    # type: int
    answers["(c) Circle (b)"] = 0

    # type: int
    answers["(c) Circle (c)"] = 2

    # type: explanatory string (at least four words)
    answers["(c) explain"] = "Since circles A and B are close together and quite far away from  circle C, the points from both A and B will be assigned to the  centroid that is in A. The points in C will be split between the  two centroids in C, with each centroid having 50,000 points.   Since A and B have the same number of points each, the centroid  in A will move between A and B. The centroids in C will move apart  slightly but both will remain in C, each having half of C’s points."
    return answers




# -----------------------------------------------------------
def question5():
    answers = {}


    # type: set
    answers["(a)"] = {'Group A', 'Group B'}

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "Groups A and B will be merged since they have the smallest  single link distance (between the right-most point of A  and left-most point of B), as compared to Groups A and C,  and Groups B and C"

    # type: set
    answers["(b)"] = {'Group C', 'Group A'}

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "Groups A and C will be merged since they have the smallest  complete link distance (between the right-most point of A  and the farthest point in C), as compared to the complete  link distance of Groups A and B (between the left-most point  in A and right-most point in B), and Groups B and C (between  rightmost-point in B and the farthest point in C)."
    return answers




# -----------------------------------------------------------
def question6():
    answers = {}


    # type: set
    answers["(a) core"] = {'J', 'C', 'I', 'F', 'B', 'E', 'M', 'L'}

    # type: set
    answers["(a) boundary"] = {'D', 'G'}

    # type: set
    answers["(a) noise"] = {'A', 'H'}

    # type: set
    answers["(b) cluster 1"] = {'C', 'D', 'G', 'F', 'B', 'E'}

    # type: set
    answers["(b) cluster 2"] = {'J', 'I', 'M', 'L'}

    # type: set
    answers["(b) cluster 3"] = set()

    # type: set
    answers["(b) cluster 4"] = set()

    # type: set
    answers["(c)-a core"] = {'J', 'C', 'D', 'G', 'I', 'F', 'B', 'E', 'M', 'L'}

    # type: set
    answers["(c)-a boundary"] = {'A', 'H'}

    # type: set
    answers["(c)-a noise"] = set()

    # type: set
    answers["(c)-b cluster 1"] = {'J', 'C', 'A', 'D', 'G', 'I', 'F', 'H', 'B', 'E', 'M', 'L'}

    # type: set
    answers["(c)-b cluster 2"] = {'A'}

    # type: set
    answers["(c)-b cluster 3"] = set()

    # type: set
    answers["(c)-b cluster 4"] = set()
    return answers




# -----------------------------------------------------------
def question7():
    answers = {}


    # type: string
    answers["(a)"] = "Cluster 4"

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "Cluster 4 is the most random."

    # type: string
    answers["(b)"] = "Cluster 1"

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "Cluster 4 is the least random."
    return answers




# -----------------------------------------------------------
def question8():
    answers = {}


    # type: string
    answers["(a) Matrix 1"] = "Dataset Z"

    # type: explanatory string (at least four words)
    answers["(a) explain diag entries, Matrix 1"] = "2 diagonal entries are more blue and crisp compared to the  other 2, indicating 2 clusters have better cohesion  (B and C) than the other 2 (A and D)."

    # type: explanatory string (at least four words)
    answers["(a) explain non-diag entries, Matrix 1"] = "1. Rows 1 and 3 correspond to clusters A and C. This is because  the colors of the off-diagonal entries for these two rows are  all different, indicating the different distances between cluster  A (or C)’ s distances to all other clusters (i.e: A is closest to C  (blue off-diagonal); followed by B (green off-diagonal); and is the  furthest from D (yellow off-diagonal); similar explanation for C). 2. Row 2 correspond to cluster B. Same distances to A and C (green  off-diagonal), furthest distance from A (red off-diagonal). 3. Row 4 correspond to cluster D. Same distances to A and C (yellow  off-diagonal), furthest distance from B (red off-diagonal)."

    # type: string
    answers["(a) Matrix 2"] = "Dataset X"

    # type: explanatory string (at least four words)
    answers["(a) explain diag entries, Matrix 2"] = "Two diagonal entries are more blue and crisp compared to the other two,  indicating two clusters have better cohesion (B and C) than the other  two (A and D)"

    # type: explanatory string (at least four words)
    answers["(a) explain non-diag entries, Matrix 2"] = "1. Rows with less crisp diagonal entries (rows 1 and 4) have all  different colors, indicating that all other clusters have different  distances from these clusters (e.g: Cluster A is the nearest to B,  followed by C and then D, no 2 clusters have same distance to cluster A).  2. Rows with more crisp diagonal entries have 2 same colors (other than  the diagonal), indicating that it has same distance to 2 clusters,  and is the furthest from 1 cluster (e.g: B’s distance to A and C is  similar, but is the furtherst from D)."

    # type: string
    answers["(a) Matrix 3"] = "Dataset Y"

    # type: explanatory string (at least four words)
    answers["(a) explain diag entries, Matrix 3"] = "Two diagonal entries are more blue and crisp compared to the other two,  indicating two clusters have better cohesion (B and C) than the other  two (A and D)."

    # type: explanatory string (at least four words)
    answers["(a) explain non-diag entries, Matrix 3"] = "All rows have two similar and 1 different colors off diagonals entries.  This indicates each cluster has two other clusters relatively closer  to it than the remaining 1 cluster (e.g: B is similarly close to A and  C compared to with D."

    # type: string
    answers["(b) Row 1"] = "Cluster A"

    # type: string
    answers["(b) Row 2"] = "Cluster B"

    # type: string
    answers["(b) Row 3"] = "Cluster C"

    # type: string
    answers["(b) Row 4"] = "Cluster D"

    # type: explanatory string (at least four words)
    answers["(b) Row 1 explain"] = "Diagonal entry is less crisp, meaning the cluster is less cohesive.  All off- diagonal entries have different colors, indicating all  other clusters have different distances from is (closest to B,  followed by C, and furthest from A)."

    # type: explanatory string (at least four words)
    answers["(b) Row 2 explain"] = "Diagonal entry is more crisp, indicating the cluster is cohesive.  2/3 off-diagonal entries have the same color, indicating 2 other  clusters are closer to it (A and C, eventhough the off-diagonal  indicating distances with A is less crisp), and is the furthest  from 1 other cluster (D)."

    # type: explanatory string (at least four words)
    answers["(b) Row 3 explain"] = "The explanation is simimilar to row 2."

    # type: explanatory string (at least four words)
    answers["(b) Row 4 explain"] = "The Explanation is similar to row 1 in inverted order."
    return answers




# -----------------------------------------------------------
def question9():
    answers = {}


    # type: list
    answers["(a)"] = ['Hierarchical', 'overlapping', 'partial']

    # type: list
    answers["(b)"] = ['Partitional', 'exclusive', 'complete']

    # type: list
    answers["(c)"] = ['Partitional', 'fuzzy', 'complete']

    # type: list
    answers["(d)"] = ['Hierarchical', 'overlapping', 'partial']

    # type: list
    answers["(e)"] = ['Partitional', 'Exclusive', 'partial']

    # type: explanatory string (at least four words)
    answers["(e) explain"] = "Some students in the CS dept wouldn’t have  taken the DM class and thus can’t be grouped."
    return answers




# -----------------------------------------------------------
def question10():
    answers = {}


    # type: string
    answers["(a) Figure (a)"] = "no"

    # type: string
    answers["(a) Figure (b)"] = "yes"

    # type: explanatory string (at least four words)
    answers["(a) explain"] = "DBSCAN can work only for (b) because in (b) the points in the nose,  eyes, and mouth are much closer together than the points between  these areas, and DBSCAN could distinguish these areas. For (a),  the noise is much denser than the interest patterns, so the nose,  eyes, and mouth will be eliminated by DBSCAN."

    # type: string
    answers["(b) Figure (a)"] = "no"

    # type: string
    answers["(b) Figure (b)"] = "yes"

    # type: explanatory string (at least four words)
    answers["(b) explain"] = "K-means can work for (b) as long as the number of clusters  was set to 4, although the lower density points would also  be included. K-means does not work for (a)."

    # type: string
    answers["(c)"] = "Take the reciprocal of the density as the new density and use DBSCAN."
    return answers

# --------------------------------------------------------
if __name__ == '__main__':
    answers_dict = {}
    answers_dict['question1'] = question1()
    answers_dict['question2'] = question2()
    answers_dict['question3'] = question3()
    answers_dict['question4'] = question4()
    answers_dict['question5'] = question5()
    answers_dict['question6'] = question6()
    answers_dict['question7'] = question7()
    answers_dict['question8'] = question8()
    answers_dict['question9'] = question9()
    answers_dict['question10'] = question10()
    print('end code')

    with open('answers.pkl', 'wb') as f:
        pickle.dump(answers_dict, f)
