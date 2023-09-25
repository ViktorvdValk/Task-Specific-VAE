import pickle


def save_file(dictionary, m_type, beta, gamma, latent_dim, mortality, part):
    """Save dictionary to pickle file."""
    
    file_name = (
        "log_dict3_"
        + m_type
        + "_g_"
        + str(gamma)
        + "_b_"
        + str(beta)
        + "_ld_"
        + str(latent_dim)
        + "_m_"
        + str(mortality)
        + part
        + ".pkl"
    )
    file = open(file_name, "wb")
    pickle.dump(dictionary, file)
    file.close()
