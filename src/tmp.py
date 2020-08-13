def get_patent_feature_vector(all_embeds):
    feature_vectors = {}
    for ID in all_embeds.keys():

        title = all_embeds[ID]['title']
        abstract = all_embeds[ID]['abstract']
        description = all_embeds[ID]['description']
        # title length set to 50
        if title.shape[0] < 50:
            title = np.vstack((title, np.zeros(50-title.shape[0], title.shape[1])))
        title = title[:50,:].flatten()
        title = title.reshape((1,len(title)))

        # abstract length set to 100
        if abstract.shape[0] < 100:
            abstract = np.vstack((abstract, np.zeros(100-abstract.shape[0], abstract.shape[1])))
        abstract = abtract[:100,:].flatten()
        abstract = abstract.reshape((1, len(abstract)))
        vec = np.hstack((vec, abstract))


        # description length set to 200
        if description.shape[0] < 200:
            description = np.vstack((description, np.zeros(200-description.shape[0], description.shape[1])))
        description = description[:200,:].flatten()
        description = description.reshape((1, len(description)))
        vec = np.hstack((vec, description))

        feature_vectors[ID] = vec

    return feature_vectors
