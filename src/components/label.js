import React from "react";
import PropTypes from "prop-types";

const Label = ({ text, value }) => {
  return (
    <h5 className="text-break">
      {text}: {value}
    </h5>
  );
};

Label.propTypes = {
  text: PropTypes.string,
  value: PropTypes.oneOfType([PropTypes.number, PropTypes.string])
};

export default Label;
