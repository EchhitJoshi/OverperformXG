
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages




# Generates a Feature Report against the Target
def generate_feature_report(dat,dat_dict,path = "/Users/echhitjoshi/Library/Mobile Documents/com~apple~CloudDocs/i360_modeling/ProjectFiles/i360_fed_budget/outputs/plots/"):
    '''
    dat: data to generate feature report
    dat_dict: data dictionary with type column for feature distinction
    path: folder to create the report in
    '''
    with PdfPages(path + "feature_report.pdf") as pdf:
        for col in dat_dict[dat_dict['modeling_feature']==1]['feature']:
            print(f"for feature: {col}")
            categorical_features = dat_dict[dat_dict['type'] == 'categorical']['feature'].values
            if col in categorical_features:
                print(f"feature {col} is categorical")
                # Check feature proportion and cr
                feature = dat.groupby([col]).agg(total_indvs = (col,'size'),debt_reduction_cr = ('response','mean')).reset_index()
                features_sorted = feature.sort_values('debt_reduction_cr',ascending=False)
                ordered_cats = features_sorted[col].tolist()

                fig,ax = plt.subplots(2,1,figsize = (20,10))
                sns.countplot(dat,x = col,hue = 'spendingresponse',order= ordered_cats,ax = ax[0])
                ax[0].set_title(f"CountPlot for feature: {col} ")
                
                features_sorted[col] = pd.Categorical(features_sorted[col],categories=ordered_cats,ordered=True)
                sns.lineplot(features_sorted,x = col,y = 'debt_reduction_cr',ax = ax[1])
                sns.scatterplot(features_sorted,x = col, y= 'debt_reduction_cr',size = 'total_indvs',ax = ax[1])
                ax[1].set_title(f"Debt Reduction CR for feature {col}")        
                
            else:
                # plot density
                fig, ax = plt.subplots(1,1,figsize =(20,10))
                sns.kdeplot(dat,x = col,hue = 'spendingresponse',ax=ax)
                ax.set_title(f"Density plot for {col}")

                
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)