import React from "react";
import PropTypes from "prop-types";

const Button = ({ value, loadingValue, isLoading, onClick }) => {
  const handleOnClick = e => {
    e.target.blur();
    onClick();
  };

  const content = () => {
    if (isLoading) {
      return (
        <React.Fragment>
          <span className="spinner-grow spinner-grow-sm mr-2 mb-1"></span>
          {loadingValue}
        </React.Fragment>
      );
    } else {
      return value;
    }
  };

  return (
    <button
      className="btn btn-primary btn-lg"
      type="button"
      onClick={handleOnClick}
    >
      {content()}
    </button>
  );
};

Button.propTypes = {
  value: PropTypes.string,
  loadingValue: PropTypes.string,
  isLoading: PropTypes.bool,
  onClick: PropTypes.func
};

export default Button;
