# playground_nlp

To check:
 - should uni_grams_bi_grams return set or list? i.e. do we want each occurrence many times? 
 
notes:

    -   to kill any processes running on port 5300 run:
        fuser -k 5100/tcp


to make env:

    -   any of:

        -   conda env create --prefix .env --file environment.yml
            This will create the environment here, under .env but you need to activate it
            using the full path i.e. conda activate ..../.env
            To delete:
            conda remove --prefix /home/christina/Documents/playground_nlp/.env --all

        -   conda env create --file environment.yml
            This will create the environment where all environments are and you need to 
            activate it by its name i.e. conda activate nlp
            To delete:
            conda remove --name nlp --all
    
