console.log("✅ custom.js loaded successfully");

async function sendQuery() {
  console.log("sendQuery triggered");

  const query = document.getElementById("queryInput").value;
  console.log("Query sent:", query);

  const response = await fetch("/execute-query/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: query })
  });

  const data = await response.json();
  console.log("Response received:", data);   // 👈 Check what backend sends

  // Build table for query results
  let tableHtml = "";
  if (data.result && data.result.rows && data.result.rows.length > 0) {
    tableHtml = "<table border='1' cellpadding='5'><tr>";
    data.result.columns.forEach(col => {
      tableHtml += `<th>${col}</th>`;
    });
    tableHtml += "</tr>";
    data.result.rows.forEach(row => {
      tableHtml += "<tr>";
      row.forEach(cell => {
        tableHtml += `<td>${cell}</td>`;
      });
      tableHtml += "</tr>";
    });
    tableHtml += "</table>";
  } else if (data.result && data.result.error) {
    tableHtml = `<p style="color:red;"><strong>Error:</strong> ${data.result.error}</p>`;
  } else {
    tableHtml = "<p>No rows returned.</p>";
  }

  document.getElementById("resultBox").innerHTML = `
    <p><strong>Query:</strong> ${data.query}</p>
    <p><strong>Execution Time:</strong> ${data.execution_time}s</p>
    <p><strong>Recommendation:</strong> ${data.recommendation}</p>
    <h4>Query Result:</h4>
    ${tableHtml}
  `;
}
