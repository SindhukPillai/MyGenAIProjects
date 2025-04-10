-- Create a database named DEMO_SNOWFLAKE_AI_DB if it doesn't already exist
create database if not exists DEMO_SNOWFLAKE_AI_DB;

-- Create a schema named DEMOLLM_SCHEMA within the database if it doesn't already exist
create schema if not exists DEMOLLM_SCHEMA;

-- Create a Python-based function named text_chunker to split PDF text into chunks
create or replace function text_chunker(pdf_text string)
returns table (chunk varchar)
language python
runtime_version = '3.9'
handler = 'text_chunker'
packages = ('snowflake-snowpark-python', 'langchain')
as
$$
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

class text_chunker:

    def process(self, pdf_text: str):
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1512, #Adjust this as you see fit
            chunk_overlap  = 256, #This lets text have some form of overlap. Useful for keeping chunks contextual
            length_function = len
        )
    
        chunks = text_splitter.split_text(pdf_text)
        df = pd.DataFrame(chunks, columns=['chunks'])
        
        yield from df.itertuples(index=False, name=None)
$$;

-- Create or replace a stage named docs for storing files, with encryption enabled
create or replace stage docs ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') DIRECTORY = ( ENABLE = true );

-- List the contents of the stage named docs
ls @docs;

-- Create a table named DOCS_CHUNKS_TABLE to hold metadata and text chunks from documents
create or replace TABLE DOCS_CHUNKS_TABLE ( 
    RELATIVE_PATH VARCHAR(16777216), -- Relative path to the PDF file
    SIZE NUMBER(38,0), -- Size of the PDF
    FILE_URL VARCHAR(16777216), -- URL for the PDF
    CHUNK VARCHAR(16777216), -- Piece of text
    CATEGORY VARCHAR(16777216) -- Will hold the document category to enable filtering
);

-- Insert data into DOCS_CHUNKS_TABLE by processing documents in the stage
insert into docs_chunks_table (relative_path, size, file_url, chunk)

    select relative_path, 
            size,
            file_url,             
            func.chunk as chunk
    from 
        directory(@docs),
        TABLE(text_chunker (TO_VARCHAR(SNOWFLAKE.CORTEX.PARSE_DOCUMENT(@docs,relative_path, {'mode': 'LAYOUT'})))) as func; -- Parse document and extract chunks

-- Query all rows from DOCS_CHUNKS_TABLE for verification
select * from docs_chunks_table;  


-- Create a temporary table docs_categories to hold document categories
CREATE OR REPLACE TEMPORARY TABLE docs_categories AS 
WITH unique_documents AS (
  SELECT
    DISTINCT relative_path
  FROM
    docs_chunks_table
),
docs_category_cte AS (
  SELECT
    relative_path,
      -- Use Snowflake Cortex to classify documents based on their names
    TRIM(snowflake.cortex.COMPLETE (
      'llama3-70b',
      'Given the name of the file between <file> and </file> determine if it is related to bike, car or bicycle. Use only one word <file> ' || relative_path || '</file>'
    ), '\n') AS category
  FROM
    unique_documents
)
SELECT * FROM docs_category_cte;


-- Query the temporary table docs_categories for verification
SELECT   * FROM   docs_categories;  


-- Update the category column in DOCS_CHUNKS_TABLE based on the classification in docs_categories
update docs_chunks_table 
  SET category = docs_categories.category
  from docs_categories
  where  docs_chunks_table.relative_path = docs_categories.relative_path;
  

-- Query all rows from DOCS_CHUNKS_TABLE after updating categories  
select * from docs_chunks_table;


-- Create a Cortex Search Service named DEMO_CC_SEARCH_SERVICE_CS for searching document chunks
create or replace CORTEX SEARCH SERVICE DEMO_CC_SEARCH_SERVICE_CS
ON chunk
ATTRIBUTES category
warehouse = COMPUTE_WH
TARGET_LAG = '1 minute'
as (
    select chunk,
        relative_path,
        file_url,
        category
    from docs_chunks_table
);



