def enable_plotly_in_cell(in_colab=True):
    """
    https://colab.research.google.com/notebooks/charts.ipynb#scrollTo=WWbPMtDkO4xg
    :param in_colab:  This function is only required in Google Colab. (Maybe PyCharm too?)
    """
    if in_colab:
        import IPython
        from plotly.offline import init_notebook_mode
        display(IPython.core.display.HTML('''
              <script src="/static/components/requirejs/require.js"></script>
        '''))
        init_notebook_mode(connected=False)
