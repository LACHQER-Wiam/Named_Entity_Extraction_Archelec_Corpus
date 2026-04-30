import sqlite3
from arkindex_export import open_database, Element, Metadata, Transcription, database
from arkindex_export.queries import list_children
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path("./data/raw/sciencespo-archelec-20260316-112904.sqlite")




def index_database(db_path: Path, vacuum: bool = False):
    """
    Add performance indexes to an Arkindex SQLite export.
    """

    # Initialize database connection
    open_database(db_path)

    if database.is_closed():
        database.connect()

    with database.atomic():
        # Critical for recursive CTE performance
        database.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_elementpath_parent_child
            ON element_path(parent_id, child_id);
        """)

        database.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_elementpath_child
            ON element_path(child_id);
        """)

        # Useful if filtering on Element.type
        database.execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_element_type
            ON element(type);
        """)

    if vacuum:
        print("Running VACUUM (this may take time)...")
        database.execute_sql("VACUUM;")

    database.close()

    print("Indexing completed.")


# Index the database before opening it
index_database(DB_PATH)

# load the  export
open_database(DB_PATH)
# create a folder to store the text files
TEXT_FOLDER = "./data/raw/arkindex_archelec/text_files"
output_folder = Path(TEXT_FOLDER)
output_folder.mkdir(exist_ok=True)
YEARS = ['1973', '1978', '1981', '1988', '1993']
ELECTIONS = ['legislatives', 'presidentielle']
folder_id = {}
for year in YEARS:
    folder_id[year]= {}
    for type in ELECTIONS:
        # create the folder
        year_folder = output_folder / year
        year_folder.mkdir(exist_ok=True)
        type_folder = year_folder / type
        type_folder.mkdir(exist_ok=True)

# compute some statistics
print("Number of folders", Element.select().where(Element.type == 'folder').count())
print("Number of pages:", Element.select().where(Element.type == 'page').count())

folder_id['1973']['legislatives'] = '929ef8f9-c2c9-4d87-b858-f7af790aa752'
folder_id['1978']['legislatives'] = '117d3883-f985-476e-b8d4-3732d8753d7a'
folder_id['1981']['legislatives'] = 'd51ea3db-68ee-4cc0-a87f-736ee17c5f87'
folder_id['1988']['legislatives'] = 'dfba9f5c-02de-478c-85c5-0ee780455433'
folder_id['1993']['legislatives'] = 'cf29300f-40bf-4b61-be93-6cb631be8fab'
#folder_id['1981']['presidentielle'] =  '4192aaa9-8485-433a-b0e3-559d2259e067'
folder_id['1988']['presidentielle'] = 'fd5bee0a-83e8-4bdc-aa48-52331af2e151'

for year in ['1973', '1978', '1981', '1988', '1993']:#YEARS:
    print ('year', year)
    for e_type in ELECTIONS:
        print ('elections', e_type)
        f_id = folder_id[year].get(e_type, None)
        if f_id:
            documents = list_children(f_id).where(Element.type == 'document')
            print(f_id,"Number of documents", documents.count())
            transcriptions_number = 0
            for document in tqdm(documents):
                pages = list_children(document.id).where(Element.type == 'page')
                transcriptions = ""
                for page in pages:
                    page_transcription = Transcription.select().where(Transcription.element == page.id).first()
                    if page_transcription:
                        transcriptions += page_transcription.text

                if transcriptions:
                    with open(f"{TEXT_FOLDER}/{year}/{e_type}/{document.name}.txt", "w") as f:
                        f.write(transcriptions)
                    transcriptions_number += 1
            print("Number of transcriptions", transcriptions_number)


