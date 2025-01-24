with employee_sales as (
    select
        e.employeeid,
        e.firstname || ' ' || e.lastname as employee_name,
        strftime('%Y', o.orderdate) as sales_year,
        sum(od.quantity * od.unitprice) as total_sales,
        count(distinct o.orderid) as total_orders
    from {{ source('northwind', 'employees') }} e
    join {{ source('northwind', 'orders') }} o on e.employeeid = o.employeeid
    join {{ source('northwind', 'order_details') }} od on o.orderid = od.orderid
    group by e.employeeid, e.firstname, e.lastname, strftime('%Y', o.orderdate)
)
--ranked_employees
    select
        *,
        rank() over (partition by sales_year order by total_sales desc) as yearly_rank
    from employee_sales

