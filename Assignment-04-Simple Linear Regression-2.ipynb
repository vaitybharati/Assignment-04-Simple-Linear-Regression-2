{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment-04-Simple Linear Regression-2"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "Q2) Salary_hike -> Build a prediction model for Salary_hike\n",
    "Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# impoort libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import statsmodels.formula.api as smf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.1</td>\n",
       "      <td>39343.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.3</td>\n",
       "      <td>46205.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>37731.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.0</td>\n",
       "      <td>43525.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.2</td>\n",
       "      <td>39891.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2.9</td>\n",
       "      <td>56642.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>3.0</td>\n",
       "      <td>60150.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3.2</td>\n",
       "      <td>54445.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3.2</td>\n",
       "      <td>64445.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>3.7</td>\n",
       "      <td>57189.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>3.9</td>\n",
       "      <td>63218.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>4.0</td>\n",
       "      <td>55794.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.0</td>\n",
       "      <td>56957.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>4.1</td>\n",
       "      <td>57081.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>4.5</td>\n",
       "      <td>61111.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>4.9</td>\n",
       "      <td>67938.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>5.1</td>\n",
       "      <td>66029.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>5.3</td>\n",
       "      <td>83088.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>5.9</td>\n",
       "      <td>81363.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>6.0</td>\n",
       "      <td>93940.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>6.8</td>\n",
       "      <td>91738.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>7.1</td>\n",
       "      <td>98273.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>7.9</td>\n",
       "      <td>101302.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>8.2</td>\n",
       "      <td>113812.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>8.7</td>\n",
       "      <td>109431.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>9.0</td>\n",
       "      <td>105582.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>9.5</td>\n",
       "      <td>116969.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>9.6</td>\n",
       "      <td>112635.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>10.3</td>\n",
       "      <td>122391.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>10.5</td>\n",
       "      <td>121872.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    YearsExperience    Salary\n",
       "0               1.1   39343.0\n",
       "1               1.3   46205.0\n",
       "2               1.5   37731.0\n",
       "3               2.0   43525.0\n",
       "4               2.2   39891.0\n",
       "5               2.9   56642.0\n",
       "6               3.0   60150.0\n",
       "7               3.2   54445.0\n",
       "8               3.2   64445.0\n",
       "9               3.7   57189.0\n",
       "10              3.9   63218.0\n",
       "11              4.0   55794.0\n",
       "12              4.0   56957.0\n",
       "13              4.1   57081.0\n",
       "14              4.5   61111.0\n",
       "15              4.9   67938.0\n",
       "16              5.1   66029.0\n",
       "17              5.3   83088.0\n",
       "18              5.9   81363.0\n",
       "19              6.0   93940.0\n",
       "20              6.8   91738.0\n",
       "21              7.1   98273.0\n",
       "22              7.9  101302.0\n",
       "23              8.2  113812.0\n",
       "24              8.7  109431.0\n",
       "25              9.0  105582.0\n",
       "26              9.5  116969.0\n",
       "27              9.6  112635.0\n",
       "28             10.3  122391.0\n",
       "29             10.5  121872.0"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# import dataset\n",
    "dataset=pd.read_csv('Database/Salary_Data.csv')\n",
    "dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EDA and Data Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 30 entries, 0 to 29\n",
      "Data columns (total 2 columns):\n",
      " #   Column           Non-Null Count  Dtype  \n",
      "---  ------           --------------  -----  \n",
      " 0   YearsExperience  30 non-null     float64\n",
      " 1   Salary           30 non-null     float64\n",
      "dtypes: float64(2)\n",
      "memory usage: 608.0 bytes\n"
     ]
    }
   ],
   "source": [
    "dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Hitesh Koli\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1935586b310>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(dataset['YearsExperience'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Hitesh Koli\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1935601a8b0>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(dataset['Salary'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>YearsExperience</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.978242</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Salary</th>\n",
       "      <td>0.978242</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 YearsExperience    Salary\n",
       "YearsExperience         1.000000  0.978242\n",
       "Salary                  0.978242  1.000000"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x19358112580>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=smf.ols(\"Salary~YearsExperience\",data=dataset).fit()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Intercept          25792.200199\n",
       "YearsExperience     9449.962321\n",
       "dtype: float64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding Cefficient Parameters\n",
    "model.params"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(Intercept          11.346940\n",
       " YearsExperience    24.950094\n",
       " dtype: float64,\n",
       " Intercept          5.511950e-12\n",
       " YearsExperience    1.143068e-20\n",
       " dtype: float64)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding Pvalues and tvalues\n",
    "model.tvalues, model.pvalues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9569566641435086, 0.9554194021486339)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding Rsquared values\n",
    "model.rsquared , model.rsquared_adj"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54142.087162"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Manual prediction for say 3 Years Experience\n",
    "Salary = (25792.200199) + (9449.962321)*(3)\n",
    "Salary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Automatic Prediction for say 3 & 5 Years Experience "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    3\n",
       "1    5\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data=pd.Series([3,5])\n",
    "new_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>YearsExperience</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   YearsExperience\n",
       "0                3\n",
       "1                5"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])\n",
    "data_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    54142.087163\n",
       "1    73042.011806\n",
       "dtype: float64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(data_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
