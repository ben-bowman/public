with customer_orders as (
    select
        c.customerid,
        c.companyname,
        sum(od.quantity * od.unitprice) as total_sales,
        avg(od.quantity * od.unitprice) as avg_order_value,
        count(distinct o.orderid) as total_orders
    from {{ source('northwind', 'customers') }} c
    join {{ source('northwind', 'orders') }} o on c.customerid = o.customerid
    join {{ source('northwind', 'order_details') }} od on o.orderid = od.orderid
    group by c.customerid, c.companyname
)
-- ranked_customers
    select
        *,
        rank() over (order by total_sales desc) as sales_rank
    from customer_orders

