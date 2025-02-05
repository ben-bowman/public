with customer_orders as (
    select
        c.customerid,
        c.companyname,
        sum(od.quantity * od.unitprice) as gross_sales,
        avg(od.quantity * od.unitprice) as avg_gross_order_value,
        sum((od.UnitPrice * od.Quantity * (1-od.Discount)) AS net_sales,
        avg((od.UnitPrice * od.Quantity * (1-od.Discount)) as avg_net_order_value,
        count(distinct o.orderid) as total_orders
    from {{ source('northwind', 'customers') }} c
    join {{ source('northwind', 'orders') }} o on c.customerid = o.customerid
    join {{ source('northwind', 'order_details') }} od on o.orderid = od.orderid
    group by c.customerid, c.companyname
)
-- ranked_customers
    select
        *,
        rank() over (order by gross_sales desc) as gross_sales_rank,
        rank() over (order by net_sales desc) as net_sales_rank
    from customer_orders
