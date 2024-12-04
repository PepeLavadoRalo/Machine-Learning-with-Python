# Book Recommendation System

This project implements a book recommendation system using collaborative filtering with **Nearest Neighbors** based on ratings data. It utilizes datasets containing book information and ratings from users. The recommendation engine uses cosine similarity to find books that are most similar to a given book.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data](#data)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Testing the Recommendation System](#testing-the-recommendation-system)

## Project Overview

This project uses **Python** and several popular libraries (`NumPy`, `Pandas`, `SciPy`, and `scikit-learn`) to process a dataset of books and ratings. It implements a recommendation engine that provides similar books based on user ratings. The system is based on collaborative filtering, using the **Nearest Neighbors** algorithm with cosine similarity.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

You need to have Python installed on your system, as well as the following Python libraries:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

If you don't have these libraries, you can install them using `pip`.

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system
```
Install the required dependencies:
You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### Data

The data required for this project is downloaded directly from an online source.

#### Dataset:

The project uses the BX-Books.csv and BX-Book-Ratings.csv datasets, which contain information about books and user ratings.
These files are automatically downloaded during execution. If for some reason the files are missing, use the following commands to download them:

```bash
wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
unzip book-crossings.zip
```
Once downloaded and extracted, these files will be used in the code to generate book recommendations.

### How It Works
The recommendation system works by creating a matrix where each row represents a book and each column represents a user.
The values in this matrix are the ratings given by users to books. The system then uses the Nearest Neighbors algorithm to find similar books based on cosine similarity. When a user queries a book, the system returns the most similar books.

### Usage
After setting up the environment and downloading the data, you can run the script to start the recommendation system. The main steps are as follows:

*Load Data*: The system loads book data (BX-Books.csv) and rating data (BX-Book-Ratings.csv).  
*Process Data*: The data is processed to count the ratings per user and per book, filtering out books with fewer than 100 ratings and users with fewer than 200 ratings.  
*Create User-Book Matrix*: A pivot table is created with books as rows and users as columns.  
*Train the Model*: The Nearest Neighbors model is trained using the user-book matrix.  
*Get Recommendations*: The user can query a book, and the system will recommend similar books based on their ratings.

### Example usage:

```python
# Get recommendations for a specific book
recommended_books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
# Print the recommended books and their distances
print(recommended_books)
```
### Testing the Recommendation System
To test the recommendation system, run the following function to validate that the recommendations are working as expected:
```python
# Run the test for a specific book recommendation
test_book_recommendation()
```
This function checks that the correct book is returned and that the recommendations are within an acceptable distance threshold.
If the tests pass, you will see:
You passed the challenge! 🎉🎉🎉🎉🎉
Otherwise, it will prompt you to try again.

