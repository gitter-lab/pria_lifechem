import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from virtual_screening.function import reshape_data_into_2_dim
import xlwt
from xlwt import Workbook

def sample_index(label_list, target_label, number):
    index_list = []
    for i in range(len(label_list)):
        if label_list[i] == target_label:
            index_list.append(i)
    index_list = np.array(index_list)
    index_list = np.random.permutation(index_list)[:number]
    return np.array(index_list)


def get_highlighted_markdown_table(matrix, pos_num):
    N = matrix.shape[0]
    content = ''
    for i in range(N):
        row = ''
        for j in range(N):
            if i <= pos_num and j <= pos_num:
                color = '#66ff66'
            elif i > pos_num and j > pos_num:
                color = '#66ffff'
            else:
                color = '#ffff99'
            row = '{}<td bgcolor={}>{:.4f}</td>'.format(row, color, matrix[i, j])
        content = '{}<tr>{}</tr>'.format(content, row)
    content = '<table>{}</table>'.format(content)

    return content


# Get covariance matrix
def get_distance_metrics(hidden_X):
    N = hidden_X.shape[0]
    matrix = np.zeros((N, N))
    for i in range(N):
        hidden_i = reshape_data_into_2_dim(hidden_X[i])
        for j in range(N):
            hidden_j = reshape_data_into_2_dim(hidden_X[j])
            # TODO: May Generalize this
            # This is for covariance matrix
            # matrix[i,j] = np.cov([hidden_i, hidden_j])[1,0]
            # or simple using
            # hidden_covariance_matrix = np.cov(hidden_X)
            matrix[i,j] = cosine_similarity(hidden_i, hidden_j)[0][0]
    return matrix


def book_set_custom_colour(book):
    book.set_colour_RGB(33, 102, 255, 102) # pos_pos
    book.set_colour_RGB(34, 255,255,153) # pos_neg
    book.set_colour_RGB(35, 102,255,255) # neg_neg
    return


def sheet_write_matrix(sheet, matrix, pos_num):
    N = matrix.shape[0]
    for i in range(N):
        for j in range(N):
            if i <= pos_num and j <= pos_num:
                color = 33
            elif i>pos_num and j>pos_num:
                color = 34
            else:
                color = 35
            style = xlwt.easyxf('pattern: pattern solid;')
            style.pattern.pattern_fore_colour = color
            style.num_format_str = r'#,###0.00000'
            sheet.write(i, j, matrix[i,j], style)
    return