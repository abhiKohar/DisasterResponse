# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # ### 2. Merge datasets.
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned later

    # merge datasets
    df = messages.merge(categories,on=["id"])
    
    return df, categories



def clean_data(df, categories):
    # ### 3. Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. 
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.
    # create a dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(pat=";",expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    # ### 4. Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : str(x)[-1]) 
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    print(categories.head())

    # ### 5. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.
    # drop the original categories column from `df`

    df.drop("categories",axis=1,inplace=True)
    print(categories.head())
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    print(df.head())
    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)
    df.groupby("related").count()
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    # ### Remove duplicates.
    # - Check how many duplicates are in this dataset.
    # - Drop the duplicates.
    # - Confirm duplicates were removed.
#     df.drop(["id","genre","original"],axis =1,inplace = True)
    df = df[df['related'].notna()] # drop None values
    # check number of duplicates
    print(df.head())
    np.sum(df.duplicated())
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    np.sum(df.duplicated())
    
    df.reset_index(drop=True)
    
    print("FIND NAN VALS",df.info())
    return df

def save_data(df, database_filename):
    # ### 7. Save the clean dataset into an sqlite database.
    
    engine = create_engine('sqlite:///' + str (database_filename))
    df.to_sql('disaster_response_table', engine, index=False,if_exists="replace")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()