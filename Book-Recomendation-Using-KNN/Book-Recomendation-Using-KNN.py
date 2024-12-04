import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Download the dataset
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

# Unzip the dataset
!unzip book-crossings.zip

# Load books and ratings data from CSV files
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Read the books CSV file
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",  # File encoding
    sep=";",  # Column separator
    header=0,  # The first row contains headers
    names=['isbn', 'title', 'author'],  # Column names
    usecols=['isbn', 'title', 'author'],  # Select relevant columns
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'}  # Data types for columns
)

# Read the ratings CSV file
df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",  # File encoding
    sep=";",  # Column separator
    header=0,  # The first row contains headers
    names=['user', 'isbn', 'rating'],  # Column names
    usecols=['user', 'isbn', 'rating'],  # Select relevant columns
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'}  # Data types for columns
)

# Calculate the number of ratings per user and per book
user_ratings = df_ratings.groupby('user', as_index=False)['rating'].count().rename(columns={'rating': 'usercount'})
book_ratings = df_ratings.groupby('isbn', as_index=False)['rating'].count().rename(columns={'rating': 'titlecount'})

# Merge the ratings data with the user and book ratings count data
df_ratings = pd.merge(left=df_ratings, right=user_ratings, on='user')
df_ratings = pd.merge(left=df_ratings, right=book_ratings, on='isbn')

# Filter users with at least 200 ratings and books with at least 100 ratings
df = df_ratings.loc[(df_ratings['usercount'] >= 200) & (df_ratings['titlecount'] >= 100)]

# Drop the user count and book count columns as they are no longer needed after filtering
df = df.drop(['usercount', 'titlecount'], axis=1)

# Merge with the books DataFrame to get book titles
df = pd.merge(left=df, right=df_books, on='isbn')

# Remove duplicates as a user cannot rate the same book more than once
df = df.drop_duplicates(['user', 'title'])

# Create a user-book matrix (pivot table) where rows are books and columns are users
df_pivot = pd.pivot(df, values='rating', index='title', columns='user')

# Fill missing values with 0 (indicating no rating given)
df_pivot = df_pivot.fillna(0)

# Convert the dataframe to a sparse matrix format (CSR) to save memory
df_matrix = csr_matrix(df_pivot.values)

# Create and train a nearest neighbors model using 'brute' algorithm and 'cosine' distance metric
model = NearestNeighbors(algorithm='brute', metric='cosine')
model.fit(df_matrix)

# Function to get book recommendations based on a given book title
def get_recommends(book=""):
    # Filter the row for the book requested
    x = df_pivot[df_pivot.index == book]

    # Get the distances and indices of the closest books
    dist, ind = model.kneighbors(x, n_neighbors=6)

    # Initialize lists for the results
    recommended_books = []
    reco_books = []

    # Flatten the distances and indices
    dist = dist.flatten()
    ind = ind.flatten()

    # Loop through the indices of the closest books
    for i in range(len(ind)):
        if i == 0:
            # The first book in the list is the book requested
            recommended_books.append(df_pivot.index[ind[i]])
        else:
            # The subsequent recommended books
            reco_book = df_pivot.index[ind[i]]
            reco_dist = dist[i]
            reco_books.append([reco_book, reco_dist])

    # Reverse the order of recommended books as expected by the test
    reco_books = reco_books[::-1]

    # Add the recommendations to the list
    recommended_books.append(reco_books)

    return recommended_books

# Test the recommendation function with a specific book
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

# Function to test the recommendation and verify if the results are correct
def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")

    # Check if the queried book is correct
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False

    # Define expected recommended books and their distances
    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]

    # Check the recommendations
    for i in range(2):  # Check only the first two recommendations
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False

    # Print the test result
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")

# Run the test
test_book_recommendation()
