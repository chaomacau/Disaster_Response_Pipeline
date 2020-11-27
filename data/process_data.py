import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load message and categories datasets. 
    
    Args: 
    messages_filepath - path to disaster_messages.csv
    categories_filepath - path to disaster_categories.csv
    
    Return:
    df - merged dataframe of messages and categories
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    ''''
    Clean dataset for downstream processing.
    
    Args: 
    df - merged dataframe of messages and categories
    
    Return:
    df - cleaned version of the input dataframe
   
    '''
    ###Split categories into columns
    categories = df['categories'].str.split(';',expand=True)
    
    ###Get column names for categories columns 
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    categories.columns = category_colnames
    
    ###Keep only values 0 and 1 for each of the category columns
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])
        categories[column] = categories[column].astype(int)
    
    categories.loc[categories['related'] == 2,'related'] = 1
    
    ###Replace the categories column with cleaned categories columns and remove duplicates
    df = df.drop(['categories'], axis=1)
    df = df.join(categories)
    df = df.drop_duplicates()
    
    return df
        
    
def save_data(df, database_filename):
    '''
    Save dataframe into SQLite 
    
    Args: 
    df - cleaned dataframe of messages and categories
    
    Return:
    database_filename - path to the database where the data is saved to.
    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages_Data', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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