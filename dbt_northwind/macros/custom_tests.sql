{% macro primary_key_tests(column_name) %}
  - not_null:
      column_name: {{ column_name }}
  - unique:
      column_name: {{ column_name }}
{% endmacro %}
